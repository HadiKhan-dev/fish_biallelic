"""Shared founder-allele posterior at one site, without 2**m enumeration.

At one site the m free founder alleles b in {0, 1}^m are shared by every
sample. Each informative sample n contributes a positive quadratic
Q_n(b) = c_n + sum_i u_ni b_i + sum_{i<j} v_nij b_i b_j (its pair weights times
its genotype evidence, built with ``polynomials.add_dosage_term``). With
linear log-prior coefficients a (log prior odds; zero for a uniform prior),

    p(b) proportional to exp(a . b) * prod_n Q_n(b).

m <= MAX_EXACT_BITS: every configuration is enumerated (each caller's exact
code; local selection's context refit uses ``exact_feedback_site``). Above that,
the uncertain core:

1. mode: deterministic loopy belief propagation with one factor per sample (a
   factor-to-bit message is the exact expectation of the quadratic under the
   other bits' beliefs), thresholded at 1/2; steepest ascent over single and
   pair flips until none improves the log posterior, from that start and from
   its full allele complement, keeping the better optimum;
2. multimodal sites abstain: when the two starts end in different optima that
   lie within SITE_CERTAIN_NATS of each other, the posterior has at least two
   comparable basins that no single or pair flip connects (e.g. sites where
   most carriers look heterozygous, nearly invariant under complementing every
   allele), the local certainty argument does not hold, and callers release
   nothing at the site (INFO_MULTIMODAL);
3. a founder is certain when its single flip and its pair flip with every other
   founder each cost more than SITE_CERTAIN_NATS at the mode;
4. the remaining core is enumerated exactly with the certain founders at the
   mode when it has at most MAX_EXACT_BITS founders; otherwise over the Hamming
   ball around the core's mode (recentred until its centre is the ball maximum)
   of the largest radius with at most SITE_BALL_CONFIGURATIONS configurations,
   but at least SITE_BALL_MINIMUM_RADIUS;
5. certain founders take their mode allele, core founders their marginal over
   the enumerated support. In the ball, a release is withheld when the marginal
   lies within SITE_SHELL_GUARD x (weight of the outermost-shell configurations
   that flip the founder) of the release threshold (``guard_withholds``).

Exact marginals of this model are #P-hard in general: above MAX_EXACT_BITS the
result is exact over the enumerated support only. Per site the work is
O(N m^2) per BP iteration and per ascent step, O(N (m^2 + 2^10)) for a core cube
and O(N m^3) for a radius-3 ball (Hamming-ball sizes are O(c^r)); configuration
likelihoods come from per-sample local fields, O(N) per configuration of a
bounded number of flips. Context refit and completion retain sparse flip sets through
prior and marginal accumulation: O(c^3) support memory and bookkeeping for
radius 3, without a dense configuration-by-founder table. The ascent and the
recentring stop at reported backstops (SITE_ASCENT_BACKSTOP, SITE_RECENTRE_BACKSTOP).

The constants reach the kernels through their ``settings`` argument
(``settings()``, a float64 vector indexed by the SETTING_* positions): Numba's
on-disk cache does not notice a changed imported global, so a cached kernel
could otherwise keep a stale value.

Release 2.1 (noise stability, design A-1/A-2/A-7): among ascent moves and ball
recentrings the first (fixed order) within SITE_IMPROVEMENT of the best is taken, so
moves that are equal by symmetry (exchangeable founders) go by founder order; a flip,
a ball recentring and the complement start's optimum must raise the log posterior by more than
SITE_IMPROVEMENT (2^-17 nats, the ``nll`` tie quantum floor; 2.0's 1e-12 sat at
the rounding and 1e-12-noise scale, so mathematically zero gains of exchangeable
founders were taken or not by their last bits). In the context refit's carrier
support, a sample whose carrier probability is exactly 1/2 by symmetry (the refit's
founder prior is uniform) is not counted within SITE_HALF_BAND (2^-26, census-placed)
of 1/2, and ``site_release_marginals`` returns the count and the mass
separately so that a mass sitting exactly at the support threshold (a sum of halves)
is decided by the caller's conservative band.
"""
import math

import numpy as np
from numba import njit

from ...maths.ties import reverse_conventions
from .polynomials import (add_dosage_term, pair_quadratic, quadratic_values, unknown_bit_index,
                          weighted_terms)


# Model constants (base core/config.py values). At most MAX_EXACT_BITS free founder bits are
# enumerated exactly (<= 1,024 configurations). Above that, founders whose single and pair
# flips at the posterior mode all cost more than SITE_CERTAIN_NATS stay at the mode; the
# remaining core is enumerated exactly when it has at most MAX_EXACT_BITS founders, otherwise
# over the largest Hamming ball with at most SITE_BALL_CONFIGURATIONS configurations, but of
# radius at least SITE_BALL_MINIMUM_RADIUS. In that ball a release is withheld when the
# marginal is within SITE_SHELL_GUARD x (outermost-shell weight that flips the founder) of its
# release threshold. The mode comes from loopy belief propagation (damping, tolerance,
# iteration limit) and single/pair flip ascent; the ascent and ball-recentring limits are
# reported backstops.
MAX_EXACT_BITS = 10
SITE_CERTAIN_NATS = 20.0
SITE_BALL_CONFIGURATIONS = 1024
SITE_BALL_MINIMUM_RADIUS = 3
SITE_SHELL_GUARD = 3.0
SITE_BP_DAMPING = 0.5
SITE_BP_TOLERANCE = 1e-9
SITE_BP_ITERATIONS = 500
SITE_ASCENT_BACKSTOP = 50
SITE_RECENTRE_BACKSTOP = 50
# Release 2.1 tie quanta: the least log-posterior improvement of a flip, a recentring or the complement
# optimum (the ``nll`` floor of maths.ties), and the band of a carrier probability around exactly 1/2 and of a
# carrier mass around the support threshold, placed by the valley census of the context refit's carrier
# probabilities (actm3, 4,674 within 1e-3 of 1/2: 1e-9 relative GL noise moves them by <= 5.7e-11, genuine
# values are dense from 3e-6): 2^-26 = 1.5e-8, >= 200x from both.
SITE_IMPROVEMENT = 2.0 ** -17
SITE_HALF_BAND = 2.0 ** -26

# Support of a site's posterior.
METHOD_CUBE = 0       # all m <= MAX_EXACT_BITS bits enumerated
METHOD_CORE_CUBE = 1  # certain founders at the mode, core of <= MAX_EXACT_BITS enumerated
METHOD_CORE_BALL = 2  # certain founders at the mode, Hamming ball over a larger core

# Columns of the per-site integer diagnostics.
INFO_METHOD = 0
INFO_CORE = 1               # core size; -1 for METHOD_CUBE
INFO_RADIUS = 2             # ball radius; 0 otherwise
INFO_BP_NONCONVERGED = 3
INFO_ASCENT_STEPS = 4
INFO_ASCENT_BACKSTOP = 5
INFO_RECENTRES = 6
INFO_RECENTRE_BACKSTOP = 7
INFO_COMPLEMENT_MODE = 8    # the complement start reached the better optimum
INFO_MULTIMODAL = 9         # the two starts' optima differ within SITE_CERTAIN_NATS: no release
INFO_FIELDS = 10

_TINY = 1e-300

# Positions in the kernels' settings vector (``settings()``).
SETTING_EXACT_BITS = 0
SETTING_CERTAIN_NATS = 1
SETTING_BALL_CONFIGURATIONS = 2
SETTING_BALL_MINIMUM_RADIUS = 3
SETTING_SHELL_GUARD = 4
SETTING_BP_DAMPING = 5
SETTING_BP_TOLERANCE = 6
SETTING_BP_ITERATIONS = 7
SETTING_ASCENT_BACKSTOP = 8
SETTING_RECENTRE_BACKSTOP = 9
SETTING_IMPROVEMENT = 10
SETTING_HALF_BAND = 11
SETTING_REVERSE = 12      # 1.0 under maths.ties.reverse_conventions (the kernels cannot read it)


def settings():
    """float64[13]: the site-posterior constants as the kernels' settings argument, and the
    convention-reversal flag (maths.ties.reverse_conventions, release 2.2 acceptance test: the
    founder order of tied ascent moves and ball recentrings and the default state of an exact-zero
    BP marginal are conventions)."""
    return np.array([
        MAX_EXACT_BITS, SITE_CERTAIN_NATS, SITE_BALL_CONFIGURATIONS, SITE_BALL_MINIMUM_RADIUS,
        SITE_SHELL_GUARD, SITE_BP_DAMPING, SITE_BP_TOLERANCE, SITE_BP_ITERATIONS,
        SITE_ASCENT_BACKSTOP, SITE_RECENTRE_BACKSTOP, SITE_IMPROVEMENT, SITE_HALF_BAND,
        1.0 if reverse_conventions() else 0.0],
        dtype=np.float64)


@njit(cache=True)
def _feedback_evidence(gl, e):
    """e = .99 gl / sum(gl) + .01 / 3 (in place); False when the sample is uninformative
    (no likelihood mass, or flat evidence after the mixture)."""
    total = gl.sum()
    if total <= 0:
        return False
    for genotype in range(3):
        e[genotype] = .99 * gl[genotype] / total + .01 / 3
    return not np.max(e) - np.min(e) < 1e-12


@njit(cache=True)
def _feedback_quadratic(e, w, background_q, fixed, bit_index, unary, coupling):
    """One sample's quadratic over the K founder alleles with a background state.

    w: float64[K + 1, K + 1] ordered-pair weights over K founders plus the background state
    (index K, whose allele is Bernoulli(background_q) per copy, integrated per copy). ``unary``
    and ``coupling`` must be zero on entry; returns the constant.
    """
    k = len(unary)
    q = background_q
    constant = w[k, k] * ((1 - q) ** 2 * e[0] + 2 * q * (1 - q) * e[1] + q * q * e[2])
    for i in range(k):
        mass = w[i, k] + w[k, i]
        f0 = (1 - q) * e[0] + q * e[1]
        f1 = (1 - q) * e[1] + q * e[2]
        constant += mass * f0
        unary[i] += mass * (f1 - f0)
        for j in range(i, k):
            mass = w[i, j] if i == j else w[i, j] + w[j, i]
            constant = add_dosage_term(constant, unary, coupling, i, j, fixed, bit_index,
                                       mass * e[0], mass * e[1], mass * e[2])
    return constant


@njit(cache=True)
def _carrier_support(w, k, support, carried_mass, half_band):
    """Carrier support of each founder from one sample's pair weights (in place): the sample
    counts in ``support`` when it is more likely than not to carry the founder (by more than
    ``half_band``: an exact 1/2 by symmetry is not counted), and its carrier probability (a
    homozygote counts once) adds to ``carried_mass``. Returns the number of carrier decisions
    inside the band."""
    inside = 0
    for i in range(k):
        carried = w[i, :].sum() + w[:, i].sum() - w[i, i]
        carried_mass[i] += carried
        if carried > .5 + half_band:
            support[i] += 1.0
        elif carried >= .5 - half_band:
            inside += 1
    return inside


@njit(cache=True)
def _exact_feedback(gl, observed, weights, background_q, half_band):
    """``exact_feedback_site`` with the carrier count and mass separate and the number of
    carrier decisions inside the 1/2 band."""
    n, s, _ = weights.shape
    k = s - 1
    count = 1 << k
    fixed = np.zeros(k, np.int8)
    bit_index = np.arange(k)
    log_joint = np.zeros(count)
    support = np.zeros(k)
    carried_mass = np.zeros(k)
    e = np.empty(3)
    inside = 0
    for sample in range(n):
        if not observed[sample]:
            continue
        if not _feedback_evidence(gl[sample], e):
            continue
        w = weights[sample]
        unary = np.zeros(k)
        coupling = np.zeros((k, k))
        constant = _feedback_quadratic(e, w, background_q, fixed, bit_index, unary, coupling)
        inside += _carrier_support(w, k, support, carried_mass, half_band)
        values = quadratic_values(constant, unary, coupling)
        for code in range(count):
            log_joint[code] += np.log(max(values[code], 1e-300))
    probabilities = np.exp(log_joint - log_joint.max())
    probabilities /= probabilities.sum()
    q = np.zeros(k)
    for code in range(count):
        for i in range(k):
            if (code >> i) & 1:
                q[i] += probabilities[code]
    for i in range(k):
        q[i] = min(1.0, max(0.0, q[i]))
    return q, support, carried_mass, probabilities, inside


@njit(cache=True)
def exact_feedback_site(gl, observed, weights, background_q, half_band=SITE_HALF_BAND):
    """Exact uniform-prior shared K-bit posterior, background integrated per copy.

    Returns q float64[K] (posterior ALT probability), the carrier support float64[K] (the larger
    of the count of samples more likely than not carrying each founder and their summed carrier
    probability) and the posterior float64[2^K] over little-endian configurations.
    """
    q, support, carried_mass, probabilities, _ = _exact_feedback(gl, observed, weights, background_q,
                                                                 half_band)
    return q, np.maximum(support, carried_mass), probabilities


@njit(cache=True)
def _feedback_quadratics(gl, observed, weights, background_q, half_band):
    """Per-sample quadratics, carrier count and mass, and the carrier decisions inside the 1/2
    band, exactly as ``_exact_feedback`` builds them."""
    n, s, _ = weights.shape
    k = s - 1
    fixed = np.zeros(k, np.int8)
    bit_index = np.arange(k)
    constants = np.zeros(n)
    unaries = np.zeros((n, k))
    couplings = np.zeros((n, k, k))
    use = np.zeros(n, np.bool_)
    support = np.zeros(k)
    carried_mass = np.zeros(k)
    e = np.empty(3)
    inside = 0
    for sample in range(n):
        if not observed[sample]:
            continue
        if not _feedback_evidence(gl[sample], e):
            continue
        use[sample] = True
        w = weights[sample]
        constants[sample] = _feedback_quadratic(e, w, background_q, fixed, bit_index,
                                                unaries[sample], couplings[sample])
        inside += _carrier_support(w, k, support, carried_mass, half_band)
    return constants, unaries, couplings, use, support, carried_mass, inside


@njit(cache=True)
def completion_quadratics(evidence, use, weights, pairs, fixed, unknown):
    """Per-sample quadratics in the unresolved alleles over unordered pair states.

    The construction of ``polynomials.shared_sample_likelihoods``: called
    founders enter through ``fixed``; only samples with ``use`` are built.
    """
    n = len(evidence)
    bits = len(unknown)
    bit_index = unknown_bit_index(len(fixed), unknown)
    constants = np.zeros(n)
    unaries = np.zeros((n, bits))
    couplings = np.zeros((n, bits, bits))
    terms = np.empty((len(pairs), 3))
    for sample in range(n):
        if not use[sample]:
            continue
        constants[sample] = pair_quadratic(
            weighted_terms(weights[sample], evidence[sample], terms), pairs, fixed, bit_index,
            unaries[sample], couplings[sample])
    return constants, unaries, couplings


@njit(cache=True)
def _logistic(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    value = math.exp(x)
    return value / (1.0 + value)


@njit(cache=True)
def _loopy_bp(constants, unaries, couplings, use, prior, settings):
    """Belief log-odds from damped flooding loopy BP, one factor per sample.

    A sample's message to bit i is log E[Q_n | b_i = 1] - log E[Q_n | b_i = 0]
    with the other bits independent at their cavity beliefs: exact for one
    quadratic factor, O(m^2) per sample from its linear and pairwise sums.
    """
    n, m = unaries.shape
    damping = settings[SETTING_BP_DAMPING]
    tolerance = settings[SETTING_BP_TOLERANCE]
    message = np.zeros((n, m))
    total = prior.copy()
    belief = np.empty(m)
    field = np.empty(m)
    converged = False
    iterations = 0
    for iteration in range(int(settings[SETTING_BP_ITERATIONS])):
        iterations = iteration + 1
        largest = 0.0
        updated = prior.copy()
        for sample in range(n):
            if not use[sample]:
                continue
            for j in range(m):
                belief[j] = _logistic(total[j] - message[sample, j])
            linear = 0.0
            for j in range(m):
                linear += unaries[sample, j] * belief[j]
            pairwise = 0.0
            for j in range(m):
                acc = 0.0
                for other in range(m):
                    if other != j:
                        acc += couplings[sample, j, other] * belief[other]
                field[j] = acc
                pairwise += 0.5 * acc * belief[j]
            for i in range(m):
                absent = (constants[sample] + linear - unaries[sample, i] * belief[i]
                          - (pairwise - belief[i] * field[i]))
                present = absent + unaries[sample, i] + field[i]
                value = math.log(max(present, _TINY)) - math.log(max(absent, _TINY))
                value = damping * message[sample, i] + (1.0 - damping) * value
                change = abs(value - message[sample, i])
                if change > largest:
                    largest = change
                message[sample, i] = value
                updated[i] += value
        total = updated
        if largest < tolerance:
            converged = True
            break
    return total, converged, iterations


@njit(cache=True)
def _local_fields(constants, unaries, couplings, use, b, base, field):
    """base[n] = Q_n(b) and field[n, i] = Q_n(b with b_i = 1) - Q_n(b with b_i = 0)."""
    n, m = unaries.shape
    for sample in range(n):
        if not use[sample]:
            continue
        value = constants[sample]
        for i in range(m):
            acc = unaries[sample, i]
            for j in range(m):
                if j != i and b[j]:
                    acc += couplings[sample, i, j]
            field[sample, i] = acc
            if b[i]:
                value += unaries[sample, i]
                for j in range(i + 1, m):
                    if b[j]:
                        value += couplings[sample, i, j]
        base[sample] = value


@njit(cache=True)
def _flip_gains(couplings, use, prior, b, base, field, single, pair):
    """Log-posterior change of every single flip and pair flip (i < j) at b, O(N m^2).

    Flipping a set F changes Q_n by sum_{i in F} s_i h_i + sum_{i<j in F} s_i s_j v_ij,
    s_i = 1 - 2 b_i, from the local fields h at b.
    """
    n, m = field.shape
    for i in range(m):
        sign = 1.0 - 2.0 * b[i]
        single[i] = sign * prior[i]
        for j in range(i + 1, m):
            pair[i, j] = sign * prior[i] + (1.0 - 2.0 * b[j]) * prior[j]
    for sample in range(n):
        if not use[sample]:
            continue
        reference = base[sample]
        log_reference = math.log(max(reference, _TINY))
        for i in range(m):
            sign = 1.0 - 2.0 * b[i]
            flipped = reference + sign * field[sample, i]
            single[i] += math.log(max(flipped, _TINY)) - log_reference
            for j in range(i + 1, m):
                other = 1.0 - 2.0 * b[j]
                value = flipped + other * field[sample, j] + sign * other * couplings[sample, i, j]
                pair[i, j] += math.log(max(value, _TINY)) - log_reference


@njit(cache=True)
def _ascend(constants, unaries, couplings, use, prior, b, base, field, single, pair, settings):
    """Steepest single/pair-flip ascent of b in place; fields and gains end at b.

    Release 2.1: a move must gain more than SITE_IMPROVEMENT, and among the moves the first (singles
    in founder order, then pairs in (i, j) order) within SITE_IMPROVEMENT of the largest gain is taken,
    so moves whose gains are equal by symmetry (exchangeable founders) are chosen by founder order,
    never by the last bits of their gains."""
    m = len(b)
    limit = int(settings[SETTING_ASCENT_BACKSTOP])
    improvement = settings[SETTING_IMPROVEMENT]
    steps = 0
    backstop = False
    while True:
        _local_fields(constants, unaries, couplings, use, b, base, field)
        _flip_gains(couplings, use, prior, b, base, field, single, pair)
        largest = -np.inf
        for i in range(m):
            if single[i] > largest:
                largest = single[i]
        for i in range(m):
            for j in range(i + 1, m):
                if pair[i, j] > largest:
                    largest = pair[i, j]
        first = -1
        second = -1
        if largest > improvement:
            cut = largest - improvement
            # Singles before pairs (fewest flips: a stated preference); inside each, founder order
            # is a convention (reversed under settings[SETTING_REVERSE]).
            reverse = settings[SETTING_REVERSE] > 0.5
            for t in range(m):
                i = m - 1 - t if reverse else t
                if single[i] >= cut:
                    first = i
                    break
            if first < 0:
                for t in range(m):
                    i = m - 1 - t if reverse else t
                    for u in range(m):
                        j = m - 1 - u if reverse else u
                        if j > i and first < 0 and pair[i, j] >= cut:
                            first = i
                            second = j
        if first < 0:
            break
        if steps >= limit:
            backstop = True
            break
        b[first] = 1 - b[first]
        if second >= 0:
            b[second] = 1 - b[second]
        steps += 1
    return steps, backstop


@njit(cache=True)
def _locally_uncertain(single, pair, settings):
    """Founders whose single flip or some pair flip costs at most SITE_CERTAIN_NATS."""
    m = len(single)
    cost = settings[SETTING_CERTAIN_NATS]
    uncertain = np.empty(m, np.bool_)
    for i in range(m):
        uncertain[i] = not single[i] < -cost
    for i in range(m):
        for j in range(i + 1, m):
            if not pair[i, j] < -cost:
                uncertain[i] = True
                uncertain[j] = True
    return uncertain


@njit(cache=True)
def _log_posterior(base, use, prior, b):
    """sum_n log Q_n(b) + prior . b, from base[n] = Q_n(b)."""
    value = 0.0
    for sample in range(len(base)):
        if use[sample]:
            value += math.log(max(base[sample], _TINY))
    for i in range(len(b)):
        if b[i]:
            value += prior[i]
    return value


@njit(cache=True)
def _ascended(constants, unaries, couplings, use, prior, start, settings):
    """Steepest single/pair-flip ascent from ``start``: optimum, base, fields, gains, log posterior."""
    n, m = unaries.shape
    b = start.copy()
    base = np.zeros(n)
    field = np.zeros((n, m))
    single = np.empty(m)
    pair = np.zeros((m, m))
    steps, backstop = _ascend(constants, unaries, couplings, use, prior, b, base, field,
                              single, pair, settings)
    return b, base, field, single, pair, _log_posterior(base, use, prior, b), steps, backstop


@njit(cache=True)
def _core_cube(constants, unaries, couplings, use, b, core):
    """Log likelihood of every core assignment, the rest fixed at b (little-endian codes)."""
    n, m = unaries.shape
    c = len(core)
    count = 1 << c
    inside = np.zeros(m, np.bool_)
    for t in range(c):
        inside[core[t]] = True
    log_likelihood = np.zeros(count)
    unary = np.empty(c)
    coupling = np.zeros((c, c))
    for sample in range(n):
        if not use[sample]:
            continue
        constant = constants[sample]
        for i in range(m):
            if not inside[i] and b[i]:
                constant += unaries[sample, i]
                for j in range(i + 1, m):
                    if not inside[j] and b[j]:
                        constant += couplings[sample, i, j]
        for t in range(c):
            i = core[t]
            acc = unaries[sample, i]
            for j in range(m):
                if not inside[j] and b[j]:
                    acc += couplings[sample, i, j]
            unary[t] = acc
            for u in range(t + 1, c):
                value = couplings[sample, i, core[u]]
                coupling[t, u] = value
                coupling[u, t] = value
        values = quadratic_values(constant, unary, coupling)
        for code in range(count):
            log_likelihood[code] += math.log(max(values[code], _TINY))
    return log_likelihood


@njit(cache=True)
def _binomial(n, k):
    if k < 0 or k > n:
        return 0
    value = 1
    for t in range(1, k + 1):
        value = value * (n - k + t) // t
    return value


@njit(cache=True)
def ball_radius(core_size, settings):
    """Largest radius within SITE_BALL_CONFIGURATIONS, at least SITE_BALL_MINIMUM_RADIUS."""
    budget = int(settings[SETTING_BALL_CONFIGURATIONS])
    radius = 0
    size = 1
    while radius < core_size and size + _binomial(core_size, radius + 1) <= budget:
        radius += 1
        size += _binomial(core_size, radius)
    return max(radius, min(int(settings[SETTING_BALL_MINIMUM_RADIUS]), core_size))


@njit(cache=True)
def _ball_flips(core_size, radius):
    """Flip sets of the ball: the centre, then lexicographic subsets of size 1..radius."""
    count = 0
    for size in range(radius + 1):
        count += _binomial(core_size, size)
    width = max(radius, 1)
    flips = np.full((count, width), -1, np.int64)
    sizes = np.zeros(count, np.int64)
    combination = np.empty(width, np.int64)
    index = 1
    for size in range(1, radius + 1):
        for t in range(size):
            combination[t] = t
        while True:
            sizes[index] = size
            for t in range(size):
                flips[index, t] = combination[t]
            index += 1
            t = size - 1
            while t >= 0 and combination[t] == core_size - size + t:
                t -= 1
            if t < 0:
                break
            combination[t] += 1
            for u in range(t + 1, size):
                combination[u] = combination[u - 1] + 1
    return flips, sizes


@njit(cache=True)
def _ball_weights(couplings, use, prior, b, core, flips, sizes, base, field,
                  log_likelihood, log_prior):
    """Log likelihood and log prior of every ball configuration relative to its centre b."""
    n = len(base)
    count = len(sizes)
    for config in range(count):
        log_likelihood[config] = 0.0
        value = 0.0
        for a in range(sizes[config]):
            i = core[flips[config, a]]
            value += (1.0 - 2.0 * b[i]) * prior[i]
        log_prior[config] = value
    for sample in range(n):
        if not use[sample]:
            continue
        reference = base[sample]
        log_reference = math.log(max(reference, _TINY))
        for config in range(1, count):
            value = reference
            for a in range(sizes[config]):
                i = core[flips[config, a]]
                sign = 1.0 - 2.0 * b[i]
                value += sign * field[sample, i]
                for d in range(a):
                    j = core[flips[config, d]]
                    value += sign * (1.0 - 2.0 * b[j]) * couplings[sample, i, j]
            log_likelihood[config] += math.log(max(value, _TINY)) - log_reference


@njit(cache=True)
def _core_ball(constants, unaries, couplings, use, prior, b, core, base, field, settings):
    """Recentred sparse Hamming ball; b, base and field end at its centre."""
    radius = ball_radius(len(core), settings)
    limit = int(settings[SETTING_RECENTRE_BACKSTOP])
    flips, sizes = _ball_flips(len(core), radius)
    count = len(sizes)
    log_likelihood = np.empty(count)
    log_prior = np.empty(count)
    recentres = 0
    backstop = False
    improvement = settings[SETTING_IMPROVEMENT]
    while True:
        _ball_weights(couplings, use, prior, b, core, flips, sizes, base, field,
                      log_likelihood, log_prior)
        # Recentre on the first configuration (ball order) within SITE_IMPROVEMENT of the best one
        # when that beats the centre by more than SITE_IMPROVEMENT (release 2.1 tie rule).
        largest = 0.0
        for config in range(1, count):
            value = log_likelihood[config] + log_prior[config]
            if value > largest:
                largest = value
        best = 0
        if largest > improvement:
            for config in range(1, count):
                if log_likelihood[config] + log_prior[config] >= largest - improvement:
                    best = config
                    break
            if settings[SETTING_REVERSE] > 0.5:
                # Fewest flips first (ball order by size: a stated preference); inside the size,
                # the subset order is a convention: the last qualifying subset under the reversal.
                for config in range(best + 1, count):
                    if sizes[config] != sizes[best]:
                        break
                    if log_likelihood[config] + log_prior[config] >= largest - improvement:
                        best = config
        if best == 0:
            break
        if recentres >= limit:
            backstop = True
            break
        for a in range(sizes[best]):
            i = core[flips[best, a]]
            b[i] = 1 - b[i]
        recentres += 1
        _local_fields(constants, unaries, couplings, use, b, base, field)
    centre = 0.0
    for sample in range(len(base)):
        if use[sample]:
            centre += math.log(max(base[sample], _TINY))
    for config in range(count):
        log_likelihood[config] += centre
    log_prior = _ball_log_prior(b, core, flips, sizes, prior)
    return radius, flips, sizes, log_likelihood, log_prior, recentres, backstop


@njit(cache=True)
def _ball_log_prior(b, core, flips, sizes, prior):
    """Core log prior from the centre and each sparse set of flips, O(B * radius)."""
    centre = 0.0
    for t in range(len(core)):
        if b[core[t]]:
            centre += prior[core[t]]
    out = np.empty(len(sizes))
    for config in range(len(sizes)):
        value = centre
        for a in range(sizes[config]):
            i = core[flips[config, a]]
            value += (1.0 - 2.0 * b[i]) * prior[i]
        out[config] = value
    return out


@njit(cache=True)
def _ball_shell(rows, sizes, radius, b, core):
    """Core founders flipped, relative to the ball's centre b, by its outermost-shell rows."""
    shell = np.zeros(rows.shape, np.bool_)
    for row in range(len(rows)):
        if sizes[row] == radius:
            for t in range(len(core)):
                shell[row, t] = rows[row, t] != b[core[t]]
    return shell


@njit(cache=True)
def _core_log_prior(rows, core, prior):
    """prior . b over the core bits of every support row (the rest is common to all rows)."""
    out = np.zeros(len(rows))
    for row in range(len(rows)):
        for t in range(len(core)):
            if rows[row, t]:
                out[row] += prior[core[t]]
    return out


@njit(cache=True)
def _compact_core_support(constants, unaries, couplings, use, prior, settings):
    """Uncertain-core support of p(b) ~ exp(prior . b) prod_n Q_n(b) for m > MAX_EXACT_BITS.

    The cube returns dense rows; the ball returns only sparse flips and sizes.
    The unused representation is empty. Both return absolute log likelihoods,
    core log priors and base/field at b. INFO_MULTIMODAL forbids a site release.
    """
    n, m = unaries.shape
    info = np.zeros(INFO_FIELDS, np.int64)
    total, converged, _ = _loopy_bp(constants, unaries, couplings, use, prior, settings)
    if not converged:
        info[INFO_BP_NONCONVERGED] = 1
    start = np.zeros(m, np.int64)
    reverse = settings[SETTING_REVERSE] > 0.5
    for i in range(m):
        # An exactly zero BP marginal starts at REF: a default state (ALT under the reversal).
        if total[i] > 0.0 or (reverse and total[i] == 0.0):
            start[i] = 1
    # Two deterministic starts, the BP threshold and its allele complement:
    # single and pair flips cannot cross between complement basins.
    b, base, field, single, pair, value, steps, backstop = _ascended(
        constants, unaries, couplings, use, prior, start, settings)
    other, base_o, field_o, single_o, pair_o, value_o, steps_o, backstop_o = _ascended(
        constants, unaries, couplings, use, prior, 1 - start, settings)
    info[INFO_ASCENT_STEPS] = steps + steps_o
    if backstop or backstop_o:
        info[INFO_ASCENT_BACKSTOP] = 1
    if value_o > value + settings[SETTING_IMPROVEMENT]:
        # The complement start's optimum replaces the BP start's only when better by more than
        # the improvement quantum (equal optima by symmetry keep the BP start's).
        b, other = other, b
        base = base_o
        field = field_o
        single = single_o
        pair = pair_o
        value, value_o = value_o, value
        info[INFO_COMPLEMENT_MODE] = 1
    if value_o >= value - settings[SETTING_CERTAIN_NATS]:
        for i in range(m):
            if other[i] != b[i]:
                # Two comparable basins no single or pair flip connects.
                info[INFO_MULTIMODAL] = 1
                break
    uncertain = _locally_uncertain(single, pair, settings)
    c = 0
    for i in range(m):
        if uncertain[i]:
            c += 1
    core = np.empty(c, np.int64)
    position = 0
    for i in range(m):
        if uncertain[i]:
            core[position] = i
            position += 1
    info[INFO_CORE] = c
    rows = np.empty((0, 0), np.int8)
    flips = np.empty((0, 0), np.int64)
    sizes = np.empty(0, np.int64)
    if c <= settings[SETTING_EXACT_BITS]:
        info[INFO_METHOD] = METHOD_CORE_CUBE
        log_likelihood = _core_cube(constants, unaries, couplings, use, b, core)
        count = len(log_likelihood)
        rows = np.empty((count, c), np.int8)
        for code in range(count):
            for t in range(c):
                rows[code, t] = (code >> t) & 1
        log_prior = _core_log_prior(rows, core, prior)
    else:
        info[INFO_METHOD] = METHOD_CORE_BALL
        radius, flips, sizes, log_likelihood, log_prior, recentres, hit = _core_ball(
            constants, unaries, couplings, use, prior, b, core, base, field, settings)
        info[INFO_RADIUS] = radius
        info[INFO_RECENTRES] = recentres
        if hit:
            info[INFO_RECENTRE_BACKSTOP] = 1
    return b, core, rows, flips, sizes, log_likelihood, log_prior, base, field, info


@njit(cache=True)
def core_support(constants, unaries, couplings, use, prior, settings):
    """Dense diagnostic support API; production callers retain compact flip sets.

    Context refit uses _compact_core_support directly and never expands a ball.
    Returns b, core, rows, shell flags, likelihoods, core priors, base, field, info.
    """
    b, core, rows, flips, sizes, log_likelihood, log_prior, base, field, info = (
        _compact_core_support(constants, unaries, couplings, use, prior, settings))
    if info[INFO_METHOD] == METHOD_CORE_BALL:
        rows = np.empty((len(sizes), len(core)), np.int8)
        for config in range(len(sizes)):
            for t in range(len(core)):
                rows[config, t] = b[core[t]]
            for a in range(sizes[config]):
                t = flips[config, a]
                rows[config, t] = 1 - rows[config, t]
        shell = _ball_shell(rows, sizes, info[INFO_RADIUS], b, core)
    else:
        shell = np.zeros(rows.shape, np.bool_)
    return b, core, rows, shell, log_likelihood, log_prior, base, field, info


@njit(cache=True)
def support_marginals(b, core, rows, shell_flags, log_likelihood, log_prior):
    """Marginals over the support (certain founders at b) and outermost-shell weights."""
    m = len(b)
    c = len(core)
    count = len(log_likelihood)
    weight = log_likelihood + log_prior
    probability = np.exp(weight - weight.max())
    probability /= probability.sum()
    q = np.zeros(m)
    for i in range(m):
        q[i] = b[i]
    for t in range(c):
        q[core[t]] = 0.0
    shell = np.zeros(m)
    boundary = 0.0
    for row in range(count):
        outer = False
        for t in range(c):
            if rows[row, t]:
                q[core[t]] += probability[row]
            if shell_flags[row, t]:
                shell[core[t]] += probability[row]
                outer = True
        if outer:
            boundary += probability[row]
    for t in range(c):
        q[core[t]] = min(1.0, max(0.0, q[core[t]]))
    return q, shell, boundary, probability


@njit(cache=True)
def _ball_marginals(b, core, flips, sizes, radius, log_likelihood, log_prior):
    """Ball marginals and shell masses from flip sets, O(B * radius + m)."""
    weight = log_likelihood + log_prior
    probability = np.exp(weight - weight.max())
    probability /= probability.sum()
    flipped = np.zeros(len(core))
    shell = np.zeros(len(b))
    boundary = 0.0
    for row in range(len(sizes)):
        outer = sizes[row] == radius and sizes[row] > 0
        mass = probability[row]
        for a in range(sizes[row]):
            t = flips[row, a]
            flipped[t] += mass
            if outer:
                shell[core[t]] += mass
        if outer:
            boundary += mass
    q = np.empty(len(b))
    for i in range(len(b)):
        q[i] = b[i]
    for t in range(len(core)):
        i = core[t]
        # A centre-ALT founder is ALT exactly when it is not in the flip set.
        value = 1.0 - flipped[t] if b[i] else flipped[t]
        q[i] = min(1.0, max(0.0, value))
    return q, shell, boundary, probability


@njit(cache=True)
def site_marginals(gl, observed, weights, background_q, settings):
    """Context-refit site posterior over the K founder alleles of one site.

    ``gl`` float64[N, 3] genotype likelihoods of the site, ``observed`` bool[N];
    ``weights[n]`` (float64[N, K + 1, K + 1]) are sample n's ordered-pair weights over K
    founders plus the background state (index K, its alleles Bernoulli(background_q) per
    copy). ``settings`` is ``settings()``. Returns q (posterior ALT probability per
    founder), support (carrier support: the larger of the number of informative
    samples more likely than not carrying each founder and their summed carrier
    probability; callers release with at least two), shell (guard weights, zero
    outside the ball), the int64 diagnostics (INFO_*; release nothing when
    ``info[INFO_MULTIMODAL]`` is set) and the ball's outermost-shell mass.
    """
    q, count, mass, shell, info, boundary, _ = site_release_marginals(gl, observed, weights,
                                                                      background_q, settings)
    return q, np.maximum(count, mass), shell, info, boundary


@njit(cache=True)
def site_release_marginals(gl, observed, weights, background_q, settings):
    """``site_marginals`` with the carrier support split into its count (samples more likely than
    not carrying the founder, by more than the 1/2 band) and its mass (summed carrier
    probabilities), plus the number of carrier decisions inside the 1/2 band; returns (q, count,
    mass, shell, info, boundary, inside)."""
    k = weights.shape[1] - 1
    half_band = settings[SETTING_HALF_BAND]
    info = np.zeros(INFO_FIELDS, np.int64)
    if k <= settings[SETTING_EXACT_BITS]:
        q, count, mass, _, inside = _exact_feedback(gl, observed, weights, background_q, half_band)
        info[INFO_METHOD] = METHOD_CUBE
        info[INFO_CORE] = -1
        return q, count, mass, np.zeros(k), info, 0.0, inside
    constants, unaries, couplings, use, count, mass, inside = _feedback_quadratics(
        gl, observed, weights, background_q, half_band)
    b, core, rows, flips, sizes, log_likelihood, log_prior, _, _, info = (
        _compact_core_support(constants, unaries, couplings, use, np.zeros(k), settings))
    if info[INFO_METHOD] == METHOD_CORE_BALL:
        q, shell, boundary, _ = _ball_marginals(
            b, core, flips, sizes, info[INFO_RADIUS], log_likelihood, log_prior)
    else:
        q, shell, boundary, _ = support_marginals(
            b, core, rows, np.zeros(rows.shape, np.bool_), log_likelihood, log_prior)
    return q, count, mass, shell, info, boundary, inside


@njit(cache=True)
def completion_site(evidence, use, weights, pairs, fixed, unknown, prior, sensitivity_tolerance,
                    settings):
    """Uncertain-core support over the U > MAX_EXACT_BITS unresolved founders of a site.

    ``weights`` are per-sample probabilities of the unordered ``pairs``,
    ``fixed`` the called alleles and ``prior`` the log prior odds of the
    unresolved founders. Returns the mode, ordered uncertain core, sparse ball
    flips and sizes (empty for a bounded cube), each configuration's log
    likelihood sum_{n in use} log Q_n, diagnostics, and, when ``sensitivity_tolerance >= 0``, which used
    samples' log likelihood changes by more than it when one founder flips
    somewhere in the support. Every support row differs from the mode only in
    core bits and Q_n is affine in each bit, so the mode and its single core
    flips decide that.
    """
    constants, unaries, couplings = completion_quadratics(
        evidence, use, weights, pairs, fixed, unknown)
    b, core, _, flips, sizes, log_likelihood, _, base, field, info = _compact_core_support(
        constants, unaries, couplings, use, prior, settings)
    bits = len(unknown)
    c = len(core)
    n = len(evidence)
    sensitive = np.zeros((bits, n), np.bool_)
    if sensitivity_tolerance >= 0.0:
        for sample in range(n):
            if not use[sample]:
                continue
            reference = base[sample]
            for i in range(bits):
                sign = 1.0 - 2.0 * b[i]
                change = (math.log(max(reference + sign * field[sample, i], _TINY))
                          - math.log(max(reference, _TINY)))
                if abs(change) > sensitivity_tolerance:
                    sensitive[i, sample] = True
                    continue
                for t in range(c):
                    j = core[t]
                    if j == i:
                        continue
                    other = 1.0 - 2.0 * b[j]
                    moved = reference + other * field[sample, j]
                    flipped = moved + sign * field[sample, i] + sign * other * couplings[sample, i, j]
                    change = math.log(max(flipped, _TINY)) - math.log(max(moved, _TINY))
                    if abs(change) > sensitivity_tolerance:
                        sensitive[i, sample] = True
                        break
    return b, core, flips, sizes, log_likelihood, info, sensitive


@njit(cache=True)
def guard_withholds(q, shell, threshold, settings):
    """Ball shell guard: a release at ``threshold`` within SITE_SHELL_GUARD x shell of it."""
    if shell <= 0.0:
        return False
    guard = settings[SETTING_SHELL_GUARD]
    if q >= threshold:
        return q - guard * shell < threshold
    if 1.0 - q >= threshold:
        return (1.0 - q) - guard * shell < threshold
    return False


def guard_mask(q, shell, threshold, settings):
    """``guard_withholds`` over arrays (bool, the shape of q)."""
    q = np.asarray(q, dtype=np.float64)
    shell = np.asarray(shell, dtype=np.float64)
    guard = settings[SETTING_SHELL_GUARD]
    alternate = (q >= threshold) & (q - guard * shell < threshold)
    reference = (1.0 - q >= threshold) & ((1.0 - q) - guard * shell < threshold)
    return (shell > 0.0) & (alternate | reference)


def shell_weights(probabilities, shell_flags):
    """Outermost-shell weight per founder column: the probability of the shell rows flipping it."""
    return np.asarray(probabilities, dtype=np.float64) @ np.asarray(shell_flags, dtype=np.float64)


SUMMARY_FIELDS = ("core_cube_sites", "core_ball_sites", "bp_nonconverged",
                  "ascent_backstop_hits", "recentre_backstop_hits", "complement_mode_sites",
                  "multimodal_abstained")


def summary_counts(info):
    """Per-site diagnostics (INFO_FIELDS x sites) -> counts in SUMMARY_FIELDS order."""
    info = np.asarray(info, dtype=np.int64).reshape(INFO_FIELDS, -1)
    method = info[INFO_METHOD]
    return (int(np.sum(method == METHOD_CORE_CUBE)), int(np.sum(method == METHOD_CORE_BALL)),
            int(np.sum(info[INFO_BP_NONCONVERGED])), int(np.sum(info[INFO_ASCENT_BACKSTOP])),
            int(np.sum(info[INFO_RECENTRE_BACKSTOP])), int(np.sum(info[INFO_COMPLEMENT_MODE])),
            int(np.sum(info[INFO_MULTIMODAL])))


def summarize_sites(info, boundary, withheld):
    """Per-component report of a context refit (sites by method, core sizes, backstops).

    info: int64[INFO_FIELDS, L] per-site diagnostics; boundary: float64[L] ball outermost-shell
    mass per site; withheld: int[L] guard-withheld releases per site.
    """
    info = np.asarray(info, dtype=np.int64)
    method = info[INFO_METHOD]
    core = info[INFO_CORE][method != METHOD_CUBE]
    ball = method == METHOD_CORE_BALL
    sizes, counts = np.unique(core, return_counts=True)
    return dict(
        cube_sites=int(np.sum(method == METHOD_CUBE)),
        core_cube_sites=int(np.sum(method == METHOD_CORE_CUBE)),
        core_ball_sites=int(np.sum(ball)),
        core_size_histogram={int(size): int(count) for size, count in zip(sizes, counts)},
        bp_nonconverged=int(np.sum(info[INFO_BP_NONCONVERGED])),
        ascent_backstop_hits=int(np.sum(info[INFO_ASCENT_BACKSTOP])),
        recentre_backstop_hits=int(np.sum(info[INFO_RECENTRE_BACKSTOP])),
        complement_mode_sites=int(np.sum(info[INFO_COMPLEMENT_MODE])),
        multimodal_abstained=int(np.sum(info[INFO_MULTIMODAL])),
        guard_withheld=int(np.sum(withheld)),
        maximum_ball_boundary_mass=float(np.max(boundary[ball])) if np.any(ball) else 0.0,
    )
