"""recombination / model for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import asdict, dataclass, replace
import math
import time
import numpy as np
from numba import njit, prange

from types import SimpleNamespace


@dataclass(frozen=True)
class RecombinationMapConfig:
    recombination_rate: float = 5e-8
    copy_error: float = .01
    phase_artifact_rate: float = 1e-8
    phase_artifact_mean_bp: float = 50_000.
    minimum_origin_probability: float = .98
    maximum_gap_bp: int = 1_000_000
    bin_bp: int = 1_000_000

    def validated(self):
        if not (np.isfinite(self.recombination_rate) and self.recombination_rate >= 0):
            raise ValueError("recombination_rate must be finite and nonnegative")
        if not 0 < self.copy_error < .5 or not .5 < self.minimum_origin_probability < 1:
            raise ValueError("invalid copy-error/origin-support probability")
        if (not np.isfinite(self.phase_artifact_rate) or self.phase_artifact_rate < 0
                or not np.isfinite(self.phase_artifact_mean_bp) or self.phase_artifact_mean_bp <= 0):
            raise ValueError("invalid correlated orientation-error process")
        if self.maximum_gap_bp <= 0 or self.bin_bp <= 0:
            raise ValueError("gap and physical bin widths must be positive")
        return self


@dataclass(frozen=True)
class SharedOrientationConfig:
    """Screening/optimization controls; rates are per physical bp."""
    phase_error_rate: float = 1e-8
    phase_error_mean_bp: float = 50_000.
    maximum_candidate_bp: int = 1_000_000
    minimum_incident_edges: int = 2
    maximum_sweeps: int = 4
    minimum_objective_gain: float = 2.
    candidate_support: float = .9

    def validated(self):
        if not np.isfinite(self.phase_error_rate) or self.phase_error_rate <= 0:
            raise ValueError("shared phase-error rate must be finite and positive")
        if not np.isfinite(self.phase_error_mean_bp) or self.phase_error_mean_bp <= 0:
            raise ValueError("shared error mean must be finite and positive")
        if (self.maximum_candidate_bp <= 0 or self.minimum_incident_edges < 2
                or self.maximum_sweeps < 1 or self.minimum_objective_gain < 0
                or not .5 < self.candidate_support < 1):
            raise ValueError("invalid shared orientation screening controls")
        return self


@njit(cache=True)
def origin_posterior(observations, positions, resets, rate, error):
    """Exact scaled forward/backward, independently initialized at resets.

observations[i] is the homologue matching the child's allele, not a REF/ALT
genotype. Missing/uninformative sites have already been marginalized out.
"""
    n = len(observations)
    forward = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        e0 = 1-error if observations[i] == 0 else error
        e1 = 1-e0
        if i == 0 or resets[i]:
            a, b = .5*e0, .5*e1
        else:
            theta = -.5*math.expm1(-2*rate*(positions[i]-positions[i-1]))
            a = ((1-theta)*forward[i-1,0]+theta*forward[i-1,1])*e0
            b = (theta*forward[i-1,0]+(1-theta)*forward[i-1,1])*e1
        forward[i,0], forward[i,1] = a/(a+b), b/(a+b)
    posterior = np.empty(n, dtype=np.float64)
    b0, b1 = 1., 1.
    for i in range(n-1, -1, -1):
        a, b = forward[i,0]*b0, forward[i,1]*b1
        posterior[i] = a/(a+b)
        if i and not resets[i]:
            theta = -.5*math.expm1(-2*rate*(positions[i]-positions[i-1]))
            e0 = 1-error if observations[i] == 0 else error
            e1 = 1-e0
            a = (1-theta)*e0*b0+theta*e1*b1
            b = theta*e0*b0+(1-theta)*e1*b1
            b0, b1 = a/(a+b), b/(a+b)
        else:
            b0, b1 = 1., 1.
    return posterior


@njit(cache=True)
def _transition(distance, genetic_distance, artifact_rate, artifact_mean):
    theta = -.5 * math.expm1(-2 * genetic_distance)
    stationary = artifact_rate / (artifact_rate + 1 / artifact_mean)
    changed = -math.expm1(-(artifact_rate + 1 / artifact_mean) * distance)
    enter, leave = stationary * changed, (1-stationary) * changed
    matrix = np.empty((4,4))
    for a in range(4):
        for b in range(4):
            p = theta if (a >> 1) != (b >> 1) else 1-theta
            p *= ((leave if not b & 1 else 1-leave) if a & 1
                  else (enter if b & 1 else 1-enter))
            matrix[a,b] = p
    return matrix


@njit(cache=True)
def origin_with_artifacts(observations, positions, resets, rate, error, artifact_rate, artifact_mean,
                          genetic_positions_morgans=None):
    """Exact four-state selector x orientation-artifact HMM.

Physical homologue S follows the recombination process. B is an independent
asymmetric continuous-time error process, entering at artifact_rate per bp
and exiting after artifact_mean bp on average. The observed matching track is
S XOR B. This models correlated phase/painting errors rather than treating a
long run of concordant wrong labels as independent genotyping errors. No
confidence from the source family model is multiplied into these emissions.
"""
    n=len(observations);exit_rate=1/artifact_mean
    stationary=artifact_rate/(artifact_rate+exit_rate)
    forward=np.empty((n,4),dtype=np.float64)
    transition=np.empty((4,4),dtype=np.float64)
    for i in range(n):
        if i and not resets[i]:
            distance=positions[i]-positions[i-1]
            if genetic_positions_morgans is None:
                theta=-.5*math.expm1(-2*rate*distance)
            else:
                theta=-.5*math.expm1(-2*(genetic_positions_morgans[i]-genetic_positions_morgans[i-1]))
            changed=-math.expm1(-(artifact_rate+exit_rate)*distance)
            enter=stationary*changed;leave=(1-stationary)*changed
            for a in range(4):
                for b in range(4):
                    p=theta if (a>>1)!=(b>>1) else 1-theta
                    if a&1:p*=leave if not b&1 else 1-leave
                    else:p*=enter if b&1 else 1-enter
                    transition[a,b]=p
        total=0.
        for b in range(4):
            value=.5*(stationary if b&1 else 1-stationary)
            if i and not resets[i]:
                value=0.
                for a in range(4):value+=forward[i-1,a]*transition[a,b]
            value*=1-error if ((b>>1)^(b&1))==observations[i] else error
            forward[i,b]=value;total+=value
        forward[i]/=total
    origin=np.empty(n);clean=np.empty(n);back=np.ones(4);following=np.empty(4)
    switch=np.zeros(n,dtype=np.float64)
    for i in range(n-1,-1,-1):
        w0=forward[i,0]*back[0];w1=forward[i,1]*back[1]
        w2=forward[i,2]*back[2];w3=forward[i,3]*back[3]
        weight=w0+w1+w2+w3
        origin[i]=(w0+w1)/weight;clean[i]=(w0+w2)/weight
        if i and not resets[i]:
            distance=positions[i]-positions[i-1]
            if genetic_positions_morgans is None:
                theta=-.5*math.expm1(-2*rate*distance)
            else:
                theta=-.5*math.expm1(-2*(genetic_positions_morgans[i]-genetic_positions_morgans[i-1]))
            changed=-math.expm1(-(artifact_rate+exit_rate)*distance)
            enter=stationary*changed;leave=(1-stationary)*changed
            jump_weight=0.;normalizer=0.
            for a in range(4):
                total=0.
                for b in range(4):
                    p=theta if (a>>1)!=(b>>1) else 1-theta
                    if a&1:p*=leave if not b&1 else 1-leave
                    else:p*=enter if b&1 else 1-enter
                    emission=1-error if ((b>>1)^(b&1))==observations[i] else error
                    value=p*emission*back[b]
                    total+=value
                    if (a>>1)!=(b>>1):jump_weight+=forward[i-1,a]*value
                following[a]=total
                normalizer+=forward[i-1,a]*total
            switch[i]=jump_weight/normalizer
            weight=following.sum()
            for a in range(4):back[a]=following[a]/weight
        else:back[:]=1.
    return origin,clean,switch


@njit(cache=True)
def edge_log_evidence(observations, positions, resets, genetic_positions,
                      copy_error, artifact_rate, artifact_mean):
    """Four-state marginal log likelihood relative to neutral Bernoulli(.5).

The ratio gives uninformative observations likelihood one. This is needed
when flipping a child's two strands exchanges known and missing alleles:
deleting an observation must not earn the log(.5) normalization advantage.
The observation axis contains only informative selected-strand observations.
"""
    stationary = artifact_rate / (artifact_rate + 1 / artifact_mean)
    previous = np.empty(4)
    current = np.empty(4)
    logz = 0.
    for i in range(len(observations)):
        if i and not resets[i]:
            matrix = _transition(positions[i]-positions[i-1],
                genetic_positions[i]-genetic_positions[i-1], artifact_rate, artifact_mean)
        for b in range(4):
            value = .5 * (stationary if b & 1 else 1-stationary)
            if i and not resets[i]:
                value = 0.
                for a in range(4):
                    value += previous[a] * matrix[a,b]
            value *= 2 * (1-copy_error if ((b >> 1) ^ (b & 1)) == observations[i] else copy_error)
            current[b] = value
        total = current.sum()
        logz += math.log(total)
        for b in range(4):
            previous[b] = current[b] / total
    return logz


def painting_coverage(product):
    """Pack actual half-open painting coverage, retaining physical gaps.

Adjacent chunks can merge regardless of founder labels; component identities
are checked independently. Unknown founder IDs do not become known alleles.
"""
    positions = np.asarray(product["positions"])
    spans = [[] for _ in product["sample_ids"]]
    for painting in product["corrected_component_paintings"]:
        for sample in painting.samples:
            spans[sample.sample_index].extend((chunk.start, chunk.end) for chunk in sample.chunks)
    packed = []; offsets = [0]
    for sample in spans:
        merged = []
        for left, right in sorted(sample):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        for left, right in merged:
            a, b = np.searchsorted(positions, [left, right], side="left")
            if b > a:
                packed.append((a, b))
        offsets.append(len(packed))
    return np.asarray(packed, dtype=np.int64).reshape(-1,2), np.asarray(offsets, dtype=np.int64)


@njit(cache=True)
def _sample_run_ids(samples, sites, coverage, offsets, components, positions, maximum_gap):
    runs = np.full((samples,sites), -1, dtype=np.int32)
    for sample in range(samples):
        run = 0
        for region in range(offsets[sample], offsets[sample+1]):
            start, stop = coverage[region]
            previous = -1
            for j in range(start,stop):
                if components[j] < 0:
                    previous = -1
                    continue
                if (previous < 0 or components[j] != components[previous]
                        or positions[j]-positions[previous] > maximum_gap):
                    run += 1
                runs[sample,j] = run
                previous = j
    return runs


@njit(cache=True)
def _one_meiosis(calls, displayed_phase, internal_phase, positions, components,
                  coverage, offsets, parent, child, slot, rate, error, threshold, max_gap, artifact_rate, artifact_mean, bins,
                  genetic_positions_morgans=None, map_positions_bp=None, map_genetic_morgans=None):
    n = len(positions)
    indices = np.empty(n, dtype=np.int64)
    observations = np.empty(n, dtype=np.int8)
    resets = np.empty(n, dtype=np.bool_)
    p_run, c_run = offsets[parent], offsets[child]
    last_p_run = last_c_run = -1
    count = 0
    for j in range(n):
        while p_run < offsets[parent+1] and coverage[p_run,1] <= j:
            p_run += 1
        while c_run < offsets[child+1] and coverage[c_run,1] <= j:
            c_run += 1
        if (components[j] < 0 or p_run == offsets[parent+1] or c_run == offsets[child+1]
                or coverage[p_run,0] > j or coverage[c_run,0] > j):
            continue
        # Public M0/M1 tracks retain their scaffold gauge. Undo that DISPLAY
        # convention to use the finalizer's internal parental homologues.
        pf = displayed_phase[parent,j] ^ internal_phase[parent,j]
        cf = displayed_phase[child,j] ^ internal_phase[child,j]
        p0, p1 = calls[parent,j,pf], calls[parent,j,1^pf]
        c = calls[child,j,slot^cf]
        if c < 0 or p0 < 0 or p1 < 0 or p0 == p1:
            continue
        reset = count == 0
        if count:
            previous = indices[count-1]
            reset = (components[j] != components[previous] or p_run != last_p_run
                or c_run != last_c_run or positions[j]-positions[previous] > max_gap)
        indices[count] = j
        observations[count] = 0 if c == p0 else 1
        resets[count] = reset
        count += 1
        last_p_run, last_c_run = p_run, c_run
    indices = indices[:count]
    obs_pos = positions[indices]
    if genetic_positions_morgans is None:
        posterior,clean,switch = origin_with_artifacts(observations[:count], obs_pos, resets[:count],
                                               rate, error, artifact_rate, artifact_mean)
        soft_counts,soft_exposure,soft_spans = posterior_intervals(obs_pos,components[indices],
            resets[:count],switch,bins,rate)
    else:
        obs_genetic = genetic_positions_morgans[indices]
        posterior,clean,switch = origin_with_artifacts(observations[:count], obs_pos, resets[:count],
                                               rate, error, artifact_rate, artifact_mean, obs_genetic)
        soft_counts,soft_exposure,soft_spans = posterior_intervals(obs_pos,components[indices],
            resets[:count],switch,bins,rate,obs_genetic,map_positions_bp,map_genetic_morgans)
    # Typically only a few spans/crossovers survive. Grow outputs locally;
    # never allocate sample x SNP x HMM-state arrays for the whole cohort.
    events = [(0., 0., 0., 0., 0.) for _ in range(0)]
    spans = [(0., 0., 0.) for _ in range(0)]
    last = -1; supported = 0
    for i in range(count):
        if resets[i]:
            last = -1
        support = max(posterior[i], 1-posterior[i])
        if support < threshold or clean[i] < threshold:
            continue
        supported += 1
        if last >= 0 and obs_pos[i]-obs_pos[last] <= max_gap:
            left, right = float(obs_pos[last]), float(obs_pos[i])
            component = float(components[indices[i]])
            if spans and spans[-1][1] == left and spans[-1][2] == component:
                spans[-1] = (spans[-1][0], right, component)
            else:
                spans.append((left, right, component))
            if (posterior[i] >= .5) != (posterior[last] >= .5):
                events.append((left, right, max(posterior[last], 1-posterior[last]), support, component))
        last = i
    event_array = np.empty((len(events),5), dtype=np.float64)
    span_array = np.empty((len(spans),3), dtype=np.float64)
    for i in range(len(events)):
        for k in range(5):event_array[i,k] = events[i][k]
    for i in range(len(spans)):
        for k in range(3):span_array[i,k] = spans[i][k]
    return (event_array, span_array, count, supported, int(np.sum(clean < threshold)),
            soft_counts,soft_exposure,soft_spans)


@njit(cache=True)
def edge_observations(calls, orientation, positions, runs, parent, child, slot,
                      maximum_gap):
    indices = np.empty(len(positions), dtype=np.int64)
    obs = np.empty(len(positions), dtype=np.int8)
    resets = np.empty(len(positions), dtype=np.bool_)
    count = 0
    for j in range(len(positions)):
        if runs[parent,j] < 0 or runs[child,j] < 0:
            continue
        p0 = calls[parent,j,orientation[parent,j]]
        p1 = calls[parent,j,1 ^ orientation[parent,j]]
        c = calls[child,j,slot ^ orientation[child,j]]
        if p0 < 0 or p1 < 0 or p0 == p1 or c < 0:
            continue
        reset = count == 0
        if count:
            previous = indices[count-1]
            reset = (runs[parent,j] != runs[parent,previous]
                     or runs[child,j] != runs[child,previous]
                     or positions[j]-positions[previous] > maximum_gap)
        indices[count], obs[count], resets[count] = j, int(c != p0), reset
        count += 1
    return indices[:count], obs[:count], resets[:count]


@njit(cache=True, parallel=True)
def _decode_meioses(calls, displayed_phase, internal_phase, positions, components,
                   coverage, offsets, parents, children, slots, rate, error, threshold, max_gap, artifact_rate, artifact_mean, bins,
                   genetic_positions_morgans=None, map_positions_bp=None, map_genetic_morgans=None):
    n = len(parents)
    events = [np.empty((0,5), dtype=np.float64) for _ in range(n)]
    spans = [np.empty((0,3), dtype=np.float64) for _ in range(n)]
    soft_spans = [np.empty((0,3), dtype=np.float64) for _ in range(n)]
    soft_counts = np.zeros((n,len(bins)-1));soft_exposure=np.zeros_like(soft_counts)
    diagnostics = np.zeros((n,3), dtype=np.int64)
    for edge in prange(n):
        event, span, informative, supported, uncertain, sc, se, ss = _one_meiosis(calls, displayed_phase, internal_phase,
            positions, components, coverage, offsets, parents[edge], children[edge], slots[edge],
            rate, error, threshold, max_gap, artifact_rate, artifact_mean, bins,
            genetic_positions_morgans,map_positions_bp,map_genetic_morgans)
        events[edge], spans[edge] = event, span
        soft_counts[edge]=sc;soft_exposure[edge]=se;soft_spans[edge]=ss
        diagnostics[edge,0], diagnostics[edge,1] = informative, supported
        diagnostics[edge,2] = uncertain
    return events, spans, diagnostics, soft_counts, soft_exposure, soft_spans


@njit(cache=True)
def orientation_log_prior(path, positions, runs, entry_rate, mean_bp):
    """One CTMC prior per individual's path, resetting at its actual breaks."""
    stationary = entry_rate / (entry_rate + 1/mean_bp)
    value = 0.
    previous = -1
    for j in range(len(positions)):
        if runs[j] < 0:
            previous = -1
            continue
        if previous < 0 or runs[j] != runs[previous]:
            probability = stationary if path[j] else 1-stationary
        else:
            changed = -math.expm1(-(entry_rate+1/mean_bp)*(positions[j]-positions[previous]))
            enter, leave = stationary*changed, (1-stationary)*changed
            probability = ((leave if not path[j] else 1-leave) if path[previous]
                           else (enter if path[j] else 1-enter))
        value += math.log(probability)
        previous = j
    return value


def _genetic_coordinates(positions, chromosome_map, genetic_positions_morgans):
    """Resolve HMM coordinates and interpolation knots once per chromosome.

    A map retains all its knots, including hotspots between observed SNPs.
    Precomputed SNP coordinates alone imply linear interpolation between SNPs.
    """
    if chromosome_map is None and genetic_positions_morgans is None:
        return None, None, None
    if genetic_positions_morgans is None:
        if not chromosome_map.has_map:
            return None, None, None
        genetic = chromosome_map.cumulative_morgans(positions)
    else:
        genetic = np.asarray(genetic_positions_morgans,dtype=np.float64)
    if (genetic.shape != positions.shape or not np.all(np.isfinite(genetic))
            or np.any(np.diff(genetic)<0)):
        raise ValueError("genetic positions must match the SNP axis and be finite and nondecreasing")
    if chromosome_map is None:
        return genetic, np.asarray(positions,dtype=np.float64), genetic
    mapped = chromosome_map.cumulative_morgans(positions)
    if not np.allclose(np.diff(genetic),np.diff(mapped),rtol=1e-10,atol=1e-14):
        raise ValueError("precomputed genetic distances differ from the chromosome map")
    knots = np.unique(np.r_[positions[0], chromosome_map.positions_bp, positions[-1]])
    return genetic,knots,chromosome_map.cumulative_morgans(knots)


def prepare_shared_data(product, relationships, config, genetic_positions_morgans=None,
                        chromosome_map=None):
    """Apply the same final-phase gauge and observed-edge rules as Stage 12."""
    config.validated()
    if not product["phase_stable"]:
        raise ValueError("shared map requires released stable final phase")
    positions = np.asarray(product["positions"], dtype=np.int64)
    components = np.asarray(product["component_ids"], dtype=np.int32)
    phase = product["phase"]
    calls = np.asarray(phase.allele_calls, dtype=np.int8)
    names = tuple(map(str, product["sample_ids"]))
    if (len(positions) < 2 or np.any(np.diff(positions) <= 0)
            or calls.shape != (len(names),len(positions),2)
            or components.shape != positions.shape):
        raise ValueError("invalid shared-map coordinate axes")
    gauge = np.asarray(phase.phase_map) ^ np.asarray(phase.conditional_result.phase_map)
    calls = np.take_along_axis(calls, np.stack((gauge,1^gauge),axis=-1),axis=-1)
    parents, children, slots = refinement_model.pedigree_edges(relationships,names)
    coverage, offsets = painting_coverage(product)
    runs = _sample_run_ids(len(names),len(positions),coverage,offsets,components,
                          positions,config.maximum_gap_bp)
    if chromosome_map is not None:
        mapped = np.asarray(chromosome_map.cumulative_morgans(positions),dtype=float)
        if (genetic_positions_morgans is not None and
                not np.allclose(mapped,genetic_positions_morgans,rtol=1e-10,atol=1e-12)):
            raise ValueError("chromosome map and supplied genetic coordinates disagree")
        genetic_positions_morgans = mapped
    if genetic_positions_morgans is None:
        genetic = (positions-positions[0]) * config.recombination_rate
    else:
        genetic = np.asarray(genetic_positions_morgans,dtype=float)
        if (genetic.shape != positions.shape or np.any(~np.isfinite(genetic))
                or np.any(np.diff(genetic) < 0)):
            raise ValueError("genetic positions must be finite, monotone Morgans on the marker axis")
    return calls,positions,runs,parents,children,slots,genetic


def decode_meioses(calls, displayed_phase, internal_phase, positions, components,
                   coverage, offsets, parents, children, slots, rate, error, threshold, max_gap,
                   artifact_rate, artifact_mean, bins, chromosome_map=None, genetic_positions_morgans=None):
    """Parallel per-meiosis decoder; optional coordinates are in Morgans.

    Artifact durations, maximum gaps and exposure always remain in physical bp.
    Without a chromosome map, precomputed coordinates imply linear genetic
    interpolation between supplied SNP positions for within-bin event mass.
    """
    genetic,knots,knot_genetic = _genetic_coordinates(positions,chromosome_map,genetic_positions_morgans)
    if chromosome_map is not None:
        rate = chromosome_map.fallback_rate_per_bp
    return _decode_meioses(calls,displayed_phase,internal_phase,positions,components,
        coverage,offsets,parents,children,slots,rate,error,threshold,max_gap,artifact_rate,artifact_mean,bins,
        genetic,knots,knot_genetic)


def _edge_fit(data, orientation, edge, config):
    calls,positions,runs,parents,children,slots,genetic = data
    indices,obs,resets = edge_observations(calls,orientation,positions,runs,
        parents[edge],children[edge],slots[edge],config.maximum_gap_bp)
    score = edge_log_evidence(obs,positions[indices],resets,genetic[indices],
        config.copy_error,config.phase_artifact_rate,config.phase_artifact_mean_bp)
    return score,(indices,obs,resets)


@njit(cache=True)
def _add_genetic_interval(bins, totals, left, right, density, map_positions_bp, map_genetic_morgans):
    """Allocate Poisson event mass by integrated hazard inside each bin overlap."""
    index = max(0, np.searchsorted(bins, left, side="right")-1)
    while index < len(totals) and bins[index] < right:
        a,b = max(left,bins[index]), min(right,bins[index+1])
        if b>a:
            mass = (np.interp(b,map_positions_bp,map_genetic_morgans)
                    - np.interp(a,map_positions_bp,map_genetic_morgans))
            totals[index] += density*mass
        index += 1


@njit(cache=True, parallel=True)
def _score_edges(calls, orientation, positions, runs, parents, children, slots,
                 genetic, edges, maximum_gap, error, artifact_rate, artifact_mean):
    """Independent exact edge integrals; coordinate moves remain sequential."""
    scores = np.empty(len(edges))
    for i in prange(len(edges)):
        edge = edges[i]
        indices,obs,resets = edge_observations(calls,orientation,positions,runs,
            parents[edge],children[edge],slots[edge],maximum_gap)
        scores[i] = edge_log_evidence(obs,positions[indices],resets,genetic[indices],
                                     error,artifact_rate,artifact_mean)
    return scores


@njit(cache=True)
def _add_interval(bins, totals, left, right, density):
    index = max(0, np.searchsorted(bins, left, side="right")-1)
    while index < len(totals) and bins[index] < right:
        width = max(0., min(right,bins[index+1])-max(left,bins[index]))
        totals[index] += density*width
        index += 1


def _scores(data, orientation, edges, config):
    calls,positions,runs,parents,children,slots,genetic = data
    return _score_edges(calls,orientation,positions,runs,parents,children,slots,
        genetic,np.asarray(edges,dtype=np.int64),config.maximum_gap_bp,config.copy_error,
        config.phase_artifact_rate,config.phase_artifact_mean_bp)


@njit(cache=True)
def posterior_intervals(positions, components, resets, switch, bins, rate,
                        genetic_positions_morgans=None, map_positions_bp=None, map_genetic_morgans=None):
    counts=np.zeros(len(bins)-1);exposure=np.zeros_like(counts)
    spans=[(0.,0.,0.) for _ in range(0)]
    for i in range(1,len(positions)):
        if resets[i]:continue
        left,right=float(positions[i-1]),float(positions[i])
        distance=right-left
        lam = rate*distance if genetic_positions_morgans is None else genetic_positions_morgans[i]-genetic_positions_morgans[i-1]
        tangent=math.tanh(lam)
        # Poisson/Haldane endpoint parity: E[N|even]=lambda*tanh(lambda),
        # E[N|odd]=lambda/tanh(lambda). This is model-based expectation,
        # not an extra observed crossover. Event positions are uniform in
        # genetic (not physical) distance under an inhomogeneous Poisson model.
        # The odd-parity conditional mean tends to one as lambda tends to zero.
        odd_mean = lam/tangent if lam>0 else 1.
        expected=lam*tangent+switch[i]*(odd_mean-lam*tangent)
        if genetic_positions_morgans is None:
            _add_interval(bins,counts,left,right,expected/distance)
        elif lam>0:
            if map_positions_bp is None:
                _add_genetic_interval(bins,counts,left,right,expected/lam,positions,genetic_positions_morgans)
            else:
                _add_genetic_interval(bins,counts,left,right,expected/lam,map_positions_bp,map_genetic_morgans)
        _add_interval(bins,exposure,left,right,1.)
        component=float(components[i])
        if spans and spans[-1][1]==left and spans[-1][2]==component:
            spans[-1]=(spans[-1][0],right,component)
        else:spans.append((left,right,component))
    array=np.empty((len(spans),3),dtype=np.float64)
    for i in range(len(spans)):
        for k in range(3):array[i,k]=spans[i][k]
    return counts,exposure,array


def _return_tracts(indices, posterior, resets, threshold):
    """Screen supported return-to-original patterns, without choosing a cause."""
    output = []
    runs = []
    for i,probability in enumerate(posterior):
        if resets[i]:
            runs = []
        if max(probability,1-probability) < threshold:
            continue
        state = int(probability < .5)
        if runs and runs[-1][0] == state:
            runs[-1][2] = i
        else:
            runs.append([state,i,i])
            if len(runs) >= 3:
                # Candidate includes the middle supported run. Its two
                # boundaries remain marker-bracketed, not known event bases.
                left = int(indices[runs[-2][1]])
                right = int(indices[runs[-1][1]])
                output.append((left,right))
    return output


def map_bins(positions,components,bin_bp):
    lo,hi=float(positions[0]),float(positions[-1])
    bins=np.r_[np.arange(lo,hi,float(bin_bp)),hi]
    changes=np.flatnonzero(components[1:]!=components[:-1])
    return np.unique(np.r_[bins,positions[changes],positions[changes+1]])


def screen_candidates(data, orientation, config, shared_config):
    """Candidate generation is deliberately screened, not exhaustive."""
    calls,positions,runs,parents,children,slots,genetic = data
    adjacent = [[] for _ in range(len(calls))]
    for edge,(parent,child) in enumerate(zip(parents,children)):
        adjacent[parent].append(edge)
        adjacent[child].append(edge)
    candidates = set()
    for edge in range(len(parents)):
        _,(indices,obs,resets) = _edge_fit(data,orientation,edge,config)
        # Use the local input map for proposal screening as well as fitting.
        # origin_posterior takes rate*distance, so Morgans with rate=1 is exact.
        posterior = origin_posterior(obs,genetic[indices],resets,1.,config.copy_error)
        for left,right in _return_tracts(indices,posterior,resets,shared_config.candidate_support):
            if positions[right]-positions[left] > shared_config.maximum_candidate_bp:
                continue
            for sample in (parents[edge],children[edge]):
                if len(adjacent[sample]) < shared_config.minimum_incident_edges:
                    continue
                if runs[sample,left] >= 0 and runs[sample,left] == runs[sample,right]:
                    candidates.add((int(sample),left,right))
    return sorted(candidates),adjacent


def aggregate_map(positions, components, events, spans, bin_bp,
                  soft_counts=None,soft_exposure=None,soft_spans=None,
                  chromosome_map=None,genetic_positions_morgans=None):
    """Interval-censored event counts / observable meiosis-bp exposure.

Spread a crossover by genetic mass within its flanking interval for binning only
(uniform physically for the scalar model);
the actual interval remains in the event table. No events or exposures cross
an assembly break. Uncovered bins have NaN rates, not zero recombination.
"""
    genetic,knots,knot_genetic = _genetic_coordinates(positions,chromosome_map,genetic_positions_morgans)
    bins=map_bins(positions,components,bin_bp)
    exposure = np.zeros(len(bins)-1); counts = np.zeros_like(exposure)
    all_spans = []
    for event, span in zip(events, spans):
        for left, right, _, _, _ in event:
            if genetic is None:
                _add_interval(bins, counts, left, right, 1/(right-left))
            else:
                mass = np.interp(right,knots,knot_genetic)-np.interp(left,knots,knot_genetic)
                if mass>0:
                    _add_genetic_interval(bins,counts,left,right,1/mass,knots,knot_genetic)
        for left, right, _ in span:
            _add_interval(bins, exposure, left, right, 1.)
            all_spans.append((left,right))
    called_exposure=exposure.copy()
    expected=counts.copy()
    if soft_counts is not None:
        expected=soft_counts.sum(axis=0);exposure=soft_exposure.sum(axis=0)
        all_spans=[(left,right) for item in soft_spans for left,right,_ in item]
    # Union physical coverage distinguishes unobserved sequence from sparse
    # sampling of otherwise covered sequence. Cumulative distance omits gaps.
    union = []
    for left, right in sorted(all_spans):
        if union and left <= union[-1][1]:
            union[-1] = (union[-1][0], max(right,union[-1][1]))
        else:union.append((left,right))
    covered = np.zeros_like(exposure)
    for left,right in union:_add_interval(bins, covered, left, right, 1.)
    rate = np.divide(expected*1e8, exposure, out=np.full_like(counts,np.nan), where=exposure>0)
    distance = rate*covered/1e6
    return {"edges_bp": bins, "crossover_count": counts, "exposure_meiosis_bp": exposure,
        "expected_crossovers":expected,"called_exposure_meiosis_bp":called_exposure,
        "called_rate_cM_per_Mb":np.divide(counts*1e8,called_exposure,out=np.full_like(counts,np.nan),where=called_exposure>0),
        "covered_bp": covered, "effective_meioses": exposure/np.diff(bins),
        "rate_cM_per_Mb": rate, "cM_on_covered_bp": distance,
        "cumulative_observed_cM": np.r_[0.,np.cumsum(np.nan_to_num(distance,nan=0.))]}


def fit_shared_orientations(product, relationships, *, config=RecombinationMapConfig(),
                            shared_config=SharedOrientationConfig(),
                            genetic_positions_morgans=None, chromosome_map=None,
                            candidates=None):
    """Screened collapsed coordinate ascent; never alters the source product.

``candidates`` is an optional list of (sample_index, first_marker, stop_marker)
    half-open tracts for independent tiny-state reference tests. Production
    screening uses only supplied alleles, inferred edges and declared priors.
    """
    started = time.monotonic()
    shared_config.validated()
    data = prepare_shared_data(product,relationships,config,genetic_positions_morgans,chromosome_map)
    calls,positions,runs,parents,children,slots,genetic = data
    orientation = np.zeros(calls.shape[:2],dtype=np.int8)
    screened,adjacent = screen_candidates(data,orientation,config,shared_config)
    candidates = screened if candidates is None else list(candidates)
    scores = _scores(data,orientation,np.arange(len(parents)),config)
    prior = np.array([orientation_log_prior(orientation[s],positions,runs[s],
        shared_config.phase_error_rate,shared_config.phase_error_mean_bp) for s in range(len(calls))])
    initial_objective = float(scores.sum()+prior.sum())
    accepted = []
    tested = 0
    converged = False
    # Deterministic descending degree order allows well-supported shared
    # parents to resolve before the more ambiguous degree-two child cases.
    candidates.sort(key=lambda c:(-len(adjacent[c[0]]),c[0],c[1],c[2]))
    for sweep in range(shared_config.maximum_sweeps):
        changes = 0
        for sample,left,right in candidates:
            orientation[sample,left:right] ^= 1
            new_prior = orientation_log_prior(orientation[sample],positions,runs[sample],
                shared_config.phase_error_rate,shared_config.phase_error_mean_bp)
            incident = adjacent[sample]
            new_scores = _scores(data,orientation,incident,config)
            per_edge = new_scores-scores[incident]
            prior_change = float(new_prior-prior[sample])
            gain = float(per_edge.sum()+prior_change)
            tested += 1
            if gain > shared_config.minimum_objective_gain:
                scores[incident] = new_scores
                prior[sample] = new_prior
                accepted.append({"sample_index":sample,"start_index":left,"stop_index":right,
                    "start_bp":int(positions[left]),"stop_bp":int(positions[right]),
                    "sweep":sweep+1,"conditional_objective_gain":gain,
                    "conditional_log_likelihood_gain":float(per_edge.sum()),
                    "log_prior_change":prior_change,"incident_edges":list(map(int,incident)),
                    "per_edge_log_likelihood_gain":list(map(float,per_edge))})
                changes += 1
            else:
                orientation[sample,left:right] ^= 1
        if not changes:
            converged = True
            break
    corrected = np.take_along_axis(calls,
        np.stack((orientation,1^orientation),axis=-1),axis=-1)
    zero = np.zeros(calls.shape[:2],dtype=np.int8)
    corrected_product = dict(product)
    corrected_product["phase"] = SimpleNamespace(allele_calls=corrected,phase_map=zero,
        conditional_result=SimpleNamespace(phase_map=zero))
    diagnostics = {"schema":"screened-shared-orientation-pilot-v1", "config":asdict(shared_config),
        "candidate_count":len(candidates),"candidate_evaluations":tested,
        "accepted_moves":accepted,"sweeps":sweep+1,"converged_on_screened_candidates":converged,
        "initial_objective":initial_objective,"final_objective":float(scores.sum()+prior.sum()),
        "individuals_with_corrections":int(np.sum(np.any(orientation,axis=1))),
        "orientation_one_markers":int(orientation.sum()),"elapsed_seconds":time.monotonic()-started,
        "interpretation":"conditional objective margins, not Bayes factors or calibrated posterior odds; screened coordinate-MAP shared paths"}
    return corrected_product,orientation,diagnostics


def build_missing_aware_map(product, relationships, *, config=RecombinationMapConfig(),
                           chromosome_map=None,genetic_positions_morgans=None):
    if chromosome_map is not None:
        config = replace(config,recombination_rate=chromosome_map.fallback_rate_per_bp)
    config = config.validated(); started = time.monotonic()
    if not product["phase_stable"]:
        raise ValueError("recombination mapping requires a released stable final phase product")
    phase = product["phase"]
    calls = np.asarray(phase.allele_calls)
    positions = np.asarray(product["positions"], dtype=np.int64)
    components = np.asarray(product["component_ids"], dtype=np.int32)
    names = tuple(map(str,product["sample_ids"]))
    if (len(positions)<2 or np.any(np.diff(positions)<=0) or calls.shape!=(len(names),len(positions),2)
            or components.shape!=positions.shape or np.any(~np.isin(calls,(-1,0,1)))):
        raise ValueError("invalid final phase allele/coordinate axes")
    parents,children,slots = refinement_model.pedigree_edges(relationships,names)
    displayed = np.asarray(phase.phase_map,dtype=np.int8)
    internal = np.asarray(phase.conditional_result.phase_map,dtype=np.int8)
    if displayed.shape!=calls.shape[:2] or internal.shape!=displayed.shape:
        raise ValueError("final phase coordinate frames differ")
    coverage,offsets = painting_coverage(product)
    bins=map_bins(positions,components,config.bin_bp)
    events,spans,diagnostics,sc,se,ss = decode_meioses(calls,displayed,internal,positions,components,
        coverage,offsets,parents,children,slots,config.recombination_rate,config.copy_error,
        config.minimum_origin_probability,config.maximum_gap_bp,
        config.phase_artifact_rate,config.phase_artifact_mean_bp,bins,
        chromosome_map=chromosome_map,genetic_positions_morgans=genetic_positions_morgans)
    curve = aggregate_map(positions,components,events,spans,config.bin_bp,sc,se,ss,
        chromosome_map=chromosome_map,genetic_positions_morgans=genetic_positions_morgans)
    result = {"schema":"missing-aware-recombination-map-v1", "contig":product["contig"],
        "sample_ids":names, "edge_parent":parents, "edge_child":children, "edge_child_slot":slots,
        "crossover_intervals":events, "callable_spans":spans, "marker_counts":diagnostics,
        "marker_count_columns":("informative","supported_origin","orientation_artifact_uncertain"),
        "informative_spans":ss,"expected_crossovers_by_edge_bin":sc,"exposure_by_edge_bin":se,
        "map":curve, "config":asdict(config),
        "summary":{"contig":product["contig"], "meioses":len(parents),
            "observable_meioses":sum(bool(len(s)) for s in spans),
            "crossovers":sum(len(e) for e in events),
            "expected_crossovers":float(curve["expected_crossovers"].sum()),
            "orientation_artifact_uncertain_markers":int(diagnostics[:,2].sum()),
            "exposure_meiosis_bp":float(curve["exposure_meiosis_bp"].sum()),
            "covered_bp":float(curve["covered_bp"].sum()),
            "cumulative_observed_cM":float(curve["cumulative_observed_cM"][-1]),
            "uncovered_bins":int(np.sum(curve["exposure_meiosis_bp"]==0)),
            "source_posterior_converged":bool(product["source_posterior_converged"]),
            "elapsed_seconds":time.monotonic()-started},
        "interpretation":"conditional on final internal family phase and fixed inferred pedigree; not calibrated joint-pedigree uncertainty",
        "event_interpretation":"supported changes of parental homologue; even crossovers within one uninformative interval are not identifiable",
        "cumulative_interpretation":"cumulative distance over covered sequence; unknown gaps omitted, not estimated as zero"}
    if chromosome_map is not None and chromosome_map.has_map:
        result["input_recombination_map"] = chromosome_map.identity()
    elif genetic_positions_morgans is not None:
        result["input_recombination_map"] = {"format":"precomputed_snp_morgans",
            "within_snp_interpolation":"piecewise_linear"}
    return result


def build_shared_family_map(product, relationships, *, config=RecombinationMapConfig(),
                            shared_config=SharedOrientationConfig(),
                            genetic_positions_morgans=None, chromosome_map=None):
    """Shared correction followed by residual per-meiosis artifact inference.

Final crossover expectations marginalize selectors/residual artifacts only,
conditional on selected shared paths. Shared-path uncertainty is NOT included.
"""
    corrected,orientation,diagnostics = fit_shared_orientations(product,relationships,
        config=config,shared_config=shared_config,
        genetic_positions_morgans=genetic_positions_morgans,chromosome_map=chromosome_map)
    kwargs = {} if genetic_positions_morgans is None else {
        "genetic_positions_morgans":genetic_positions_morgans}
    if chromosome_map is not None:kwargs["chromosome_map"] = chromosome_map
    result = build_missing_aware_map(corrected,relationships,config=config,**kwargs)
    result["schema"] = "shared-orientation-recombination-map-v1"
    result["shared_orientation"] = diagnostics
    result["interpretation"] += "; additionally conditional on screened coordinate-MAP shared sample orientation corrections"
    result["summary"]["shared_orientation_elapsed_seconds"] = diagnostics["elapsed_seconds"]
    result["summary"]["elapsed_seconds"] += diagnostics["elapsed_seconds"]
    return result

import haplotype_reconstruction.refinement.model as refinement_model
