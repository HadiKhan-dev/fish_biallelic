"""Quadratic log-domain counterpart of the normal/error sum-product scan.

This is a numerical fallback, not a different model. Prefix/suffix log sums
exclude a homologue without subtracting a dominant term or using cubic loops.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _row_exclusions(values, out):
    for row in range(values.shape[0]):
        prefix = -np.inf
        for col in range(values.shape[1]):
            out[row,col] = prefix
            prefix = np.logaddexp(prefix,values[row,col])
        suffix = -np.inf
        for col in range(values.shape[1]-1,-1,-1):
            out[row,col] = np.logaddexp(out[row,col],suffix)
            suffix = np.logaddexp(suffix,values[row,col])


@njit(cache=True, parallel=True)
def scan_sum_product(tensor,error_logs,stay,switch,quality,priors,haps,backward,
                     emission_indices,error_fraction):
    samples,sites,_ = tensor.shape
    output = np.full((samples,haps*haps),-np.inf)
    log_stay,log_switch,log_quality = np.log(stay),np.log(switch),np.log(quality)
    lp = math.log(error_fraction) if error_fraction > 0 else -np.inf
    ln = math.log1p(-error_fraction)
    for sample in prange(samples):
        normal = np.empty((haps,haps))
        error = np.empty_like(normal)
        next_normal = np.empty_like(normal)
        next_error = np.empty_like(normal)
        xn = np.empty_like(normal)
        xe = np.empty_like(normal)
        first = sites-1 if backward else 0
        k = 0
        for a in range(haps):
            for b in range(a,haps):
                prior = priors[sample,a*haps+b]
                n = prior + np.float64(tensor[sample,first,emission_indices[first,k]])
                e = prior + error_logs[sample,first]
                if not backward:
                    n += ln
                    e += lp
                normal[a,b] = normal[b,a] = n
                error[a,b] = error[b,a] = e
                k += 1
        for step in range(1,sites):
            site = sites-1-step if backward else step
            interval = site if backward else site-1
            _row_exclusions(normal,xn)
            _row_exclusions(error,xe)
            nn,ne,en,ee = log_quality[interval]
            k = 0
            for a in range(haps):
                for b in range(a,haps):
                    hn = np.logaddexp(log_stay[interval]+normal[a,b],
                        log_switch[interval]+np.logaddexp(xn[a,b],xn[b,a]))
                    he = np.logaddexp(log_stay[interval]+error[a,b],
                        log_switch[interval]+np.logaddexp(xe[a,b],xe[b,a]))
                    if backward:
                        n = np.logaddexp(nn+hn,ne+he)
                        e = np.logaddexp(en+hn,ee+he)
                    else:
                        n = np.logaddexp(nn+hn,en+he)
                        e = np.logaddexp(ne+hn,ee+he)
                    n += np.float64(tensor[sample,site,emission_indices[site,k]])
                    e += error_logs[sample,site]
                    next_normal[a,b] = next_normal[b,a] = n
                    next_error[a,b] = next_error[b,a] = e
                    k += 1
            normal,next_normal = next_normal,normal
            error,next_error = next_error,error
        for a in range(haps):
            for b in range(a,haps):
                value = (np.logaddexp(normal[a,b]+ln,error[a,b]+lp) if backward
                         else np.logaddexp(normal[a,b],error[a,b]))
                output[sample,a*haps+b] = output[sample,b*haps+a] = value
    return output
