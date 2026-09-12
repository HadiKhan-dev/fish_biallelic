"""Fused fixed-evidence and phase-view kernels for family refinement."""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def prepare_evidence(gl, observed, reference, roots, condition, root_prior):
    samples, sites = observed.shape
    base = np.empty((samples,sites,4),dtype=np.float32)
    orientation = np.full((samples,sites),-1,dtype=np.int8)
    anchors = np.full(samples,-1,dtype=np.int64)
    invalid = np.zeros(samples,dtype=np.bool_)
    for sample in prange(samples):
        anchor = -1
        for site in range(sites):
            a,b = reference[sample,site,0],reference[sample,site,1]
            g0,g1,g2 = gl[sample,site,0],gl[sample,site,1],gl[sample,site,2]
            if (a < -1 or a > 1 or b < -1 or b > 1 or
                    not math.isfinite(g0) or not math.isfinite(g1) or not math.isfinite(g2)
                    or min(g0,g1,g2) < 0):
                invalid[sample] = True
            total = (g0+g1)+g2
            if not observed[sample,site] or total <= 0:
                p0,p1,p2 = 1/3,1/3,1/3
            else:
                p0,p1,p2 = g0/total,g1/total,g2/total
            values = (p0,p1,p1,p2)
            for state in range(4):
                dosage = (state >> 1)+(state & 1)
                allowed = True
                if condition:
                    if a >= 0 and b >= 0: allowed = dosage == a+b
                    elif a >= 0: allowed = 0 <= dosage-a <= 1
                    elif b >= 0: allowed = 0 <= dosage-b <= 1
                value = np.float32(math.log(max(values[state],1e-30))) if allowed else np.float32(-np.inf)
                if roots[sample]: value = np.float32(value+root_prior[state])
                base[sample,site,state] = value
            if a >= 0 and b >= 0 and a != b:
                orientation[sample,site] = a
                if roots[sample] and anchor < 0: anchor = site
        anchors[sample] = anchor
    return base,orientation,anchors,invalid


@njit(cache=True, parallel=True)
def phase_views(reference, emitted, initial):
    samples,sites,_ = reference.shape
    calls = np.empty_like(reference)
    provenance = np.zeros(reference.shape,dtype=np.uint8)
    changed = np.empty((samples,sites),dtype=np.bool_)
    counts = np.zeros((samples,3),dtype=np.int64)
    invalid = np.zeros(samples,dtype=np.bool_)
    for sample in prange(samples):
        for site in range(sites):
            f = emitted[sample,site]
            r0,r1 = reference[sample,site,0],reference[sample,site,1]
            changed[sample,site] = f != initial[sample,site] and r0 != r1
            for slot in range(2):
                original = reference[sample,site,slot ^ f]
                value = original
                calls[sample,site,slot] = value
                counts[sample,0] += value >= 0
                counts[sample,1] += original >= 0
                if value >= 0:
                    provenance[sample,site,slot] = 2 if original < 0 else 1
                    counts[sample,2] += original < 0
            a,b = calls[sample,site,0],calls[sample,site,1]
            if min(a,b) != min(r0,r1) or max(a,b) != max(r0,r1):
                invalid[sample] = True
    return calls,provenance,changed,counts,invalid


@njit(cache=True, parallel=True)
def valid_phase_arrays(reference, phase):
    invalid = np.zeros(len(reference),dtype=np.bool_)
    for sample in prange(len(reference)):
        for site in range(reference.shape[1]):
            a,b = reference[sample,site,0],reference[sample,site,1]
            f = phase[sample,site]
            if a < -1 or a > 1 or b < -1 or b > 1 or f < 0 or f > 1:
                invalid[sample] = True
    return not np.any(invalid)
