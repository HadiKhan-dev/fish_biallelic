"""Cohort genotype frequencies of every site: the outlier law of the observation model (4.21; science.shared.observation).

The observation model of every stage mixes a model's genotype law with the site's cohort
genotype frequencies q_s: the law of a genotype drawn from the cohort at the site, without a Hardy-Weinberg assumption
(an experimental cross is not in Hardy-Weinberg equilibrium), estimated by pseudocount-smoothed EM over the observed
samples' normalised genotype likelihoods:

    q <- (sum_n post_n + a) / (n_observed + 3 a),    post_n(g) = q(g) L_n(g) / sum_h q(h) L_n(h)

with a = COHORT_PRIOR pseudo-observations per genotype, started at the uniform law and iterated to COHORT_TOLERANCE (at
most COHORT_MAX_ITERATIONS). At its fixed point this is the penalised-likelihood (MAP) estimate under a Dirichlet(a + 1)
prior, not the exact posterior mean under Dirichlet(a): the pseudocounts keep every frequency positive (no clip) and move
a frequency by at most 2 a / (n_observed + 3 a). Unobserved cells carry no information and are left out; a site no
sample observes has the uniform law. The population is the caller's: the evidence stage fits q over the founder view
(samples the evidence QC withholds from founder construction do not contribute), and every stage reads that one stored
law, including for the withheld samples. q is an empirical-Bayes quantity that includes the scored samples: it is not an
independent null. A sample's own contribution matters on the log scale for a genotype only it carries (with sharp
likelihoods q = 1.5 / (N + 1.5) with it, 0.5 / (N + 0.5) without: about 1.1 nats of log B at N = 40); held-out
scores (the observation pilot) read this q, a disclosed leak.
"""

from __future__ import annotations

import numpy as np
from numba import types

from ...contracts.likelihoods import gather_likelihoods, likelihood_window_sites
from ...contracts.observations import gather_observed
from ...core.kernels import kernel

__all__ = ["cohort_genotype_frequencies", "COHORT_PRIOR", "COHORT_MAX_ITERATIONS", "COHORT_TOLERANCE"]

COHORT_PRIOR = 0.5
"""Pseudo-observations per genotype class added at every EM update (module docstring): no class has zero frequency, so
the outlier law needs no clip; with N observed samples a class frequency moves by at most 0.5 / (N + 1.5)."""
COHORT_MAX_ITERATIONS = 500
"""Upper bound on the EM updates of one site (iterations); the tolerance ends every fit first in practice."""
COHORT_TOLERANCE = 1e-10
"""EM stops when the L1 change of a site's three frequencies falls below this (probability)."""

_F2 = types.Array(types.float64, 2, "C")
_F3 = types.Array(types.float64, 3, "C")
_B2 = types.Array(types.boolean, 2, "C")


@kernel(overloads=((_F3, _B2, types.float64, types.int64, types.float64, _F2),), cache=True, nogil=True)
def _fit_sites(likelihoods, observed, prior, iterations, tolerance, out):
    """EM of every site's genotype frequencies (likelihoods [N, W, 3], observed [N, W]) into out [W, 3]."""
    samples, width = observed.shape
    post = np.zeros(3)
    for s in range(width):
        q0 = 1.0 / 3.0
        q1 = 1.0 / 3.0
        q2 = 1.0 / 3.0
        count = 0
        for n in range(samples):
            if observed[n, s]:
                count += 1
        if count == 0:
            out[s, 0] = q0
            out[s, 1] = q1
            out[s, 2] = q2
            continue
        for _ in range(iterations):
            post[0] = 0.0
            post[1] = 0.0
            post[2] = 0.0
            for n in range(samples):
                if not observed[n, s]:
                    continue
                l0 = likelihoods[n, s, 0]
                l1 = likelihoods[n, s, 1]
                l2 = likelihoods[n, s, 2]
                total = l0 + l1 + l2
                if total <= 0.0:
                    continue
                w0 = q0 * l0
                w1 = q1 * l1
                w2 = q2 * l2
                z = w0 + w1 + w2
                if z <= 0.0:
                    continue
                post[0] += w0 / z
                post[1] += w1 / z
                post[2] += w2 / z
            denominator = count + 3.0 * prior
            n0 = (post[0] + prior) / denominator
            n1 = (post[1] + prior) / denominator
            n2 = (post[2] + prior) / denominator
            change = abs(n0 - q0) + abs(n1 - q1) + abs(n2 - q2)
            q0 = n0
            q1 = n1
            q2 = n2
            if change < tolerance:
                break
        out[s, 0] = q0
        out[s, 1] = q1
        out[s, 2] = q2


def cohort_genotype_frequencies(likelihoods, observed, n_sites: int, *, excluded=None, map_work=map) -> np.ndarray:
    """float64 [L, 3] cohort genotype frequencies of every site (module docstring) of a dense, coded or windowed
    likelihood source and its observation flags; ``excluded`` (bool [N]) leaves samples out (the evidence stage's
    QC-flagged libraries: the cohort is the founder view). Windows of sites are independent and run on ``map_work``
    (ordered; one CPU each); the values do not depend on the window size."""
    keep = None if excluded is None or not np.any(excluded) else ~np.asarray(excluded, dtype=np.bool_)
    width = likelihood_window_sites(likelihoods) if not isinstance(likelihoods, np.ndarray) else 4096
    windows = [(start, min(n_sites, start + width)) for start in range(0, n_sites, width)]

    def window(bounds):
        start, stop = bounds
        columns = np.arange(start, stop)
        gl = np.ascontiguousarray(gather_likelihoods(likelihoods, columns), dtype=np.float64)
        seen = np.ascontiguousarray(gather_observed(observed, columns), dtype=np.bool_)
        if keep is not None:
            seen &= keep[:, None]
        out = np.empty((stop - start, 3), dtype=np.float64)
        _fit_sites(gl, seen, float(COHORT_PRIOR), int(COHORT_MAX_ITERATIONS), float(COHORT_TOLERANCE), out)
        return out

    parts = list(map_work(window, windows))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float64)
