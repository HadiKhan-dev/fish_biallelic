"""Capped empirical-Bayes per-copy uncertainty conditional on fitted carriers and panel (3.4).

The release gives each homologue a flip probability d_x relative to its selected panel allele. It removes the
sample's direct own likelihood term while holding the fitted carriers, panel and empirical prior fixed. This is
not a cross-fitted refit and does not integrate over carrier or panel selection: the sample's reads can still
influence those fitted quantities. The selected allele also determines which disagreement probability is capped.

- Carriers: the samples whose solved MAP label pair contains x and whose partner has a known allele (or is x),
  excluding founder-QC-excluded samples. Unobserved samples have flat likelihoods and add nothing.
- Each carrier adds l(g1) - l(g0), with l(g) = log((1 - eps) GL(g) + eps / 3), conditioning on the selected partner
  allele. For (x, x), the evidence compares genotypes 2 and 0 once. E(x, j) sums those terms.
- The released sample's direct term is removed from its own MAP labels; other labels contain no direct term from
  it. This subtraction does not remove its indirect influence on the same-data carrier/panel selection.
- The empirical prior is the chromosome's allele-1 share among called refit row-by-site cells, clipped to
  [0.01, 0.99]. It uses no truth or sample metadata and is not a prior estimated independently of the fitted panel.
- p = sigmoid(logit pi + E - own). Disagreement is p for a selected REF allele and 1 - p for selected ALT;
  d_x is that disagreement capped at 1/2 and set to zero below FLIP_NEGLIGIBLE. The cap depends on the selected allele.
- The site model flips the two copies independently per homologue, including (x, x). It is not exact integration
  over one shared latent allele.

Gate 1 assessed min(p, 1-p) on mapped single-row labels, whereas release uses capped disagreement with the selected
allele. Pooled and unmapped labels are unvalidated. Correlated carrier identity errors remain a failure mode;
well-carried classes were 3-9 times overconfident on four designs. These limits are separate from the improved
measured downstream genotype-score calibration.

Evidence costs O(N m) per component and solve (chunked temporaries); a sample's own terms cost O(m).
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from .diagnostics import DecisionTrace, PhaseTrace
from .parameters import EVIDENCE_CHUNK, FLIP_CEILING, FLIP_NEGLIGIBLE, LOGIT_LIMIT, PRIOR_BOUNDS


def allele_prior(ones: int, called: int) -> float:
    """CAL2 empirical logit from exact contig-wide counts; mutate nothing (P0 FP14)."""
    share = ones / called if called else 0.5
    share = min(max(share, PRIOR_BOUNDS[0]), PRIOR_BOUNDS[1])
    return float(np.log(share / (1.0 - share)))


def _robust_log(gl, eps):
    return np.log((1.0 - eps) * np.asarray(gl, dtype=np.float64) + eps / 3.0)


def _terms(lab0, lab1, alleles_0, alleles_1, lg):
    """Carrier terms of label pairs (lab0, lab1) with alleles (alleles_0, alleles_1) and robust log likelihoods lg
    [..., 3]: (e0, valid0, e1, valid1), the evidence for allele 1 of each homologue's label (a (x, x) pair counts
    once, on homologue 0)."""
    same = lab0 == lab1
    # The genotype columns of an allele a in {-1, 0, 1} clipped to {0, 1}: lg[..., a] and lg[..., a + 1], selected
    # directly (the same values as an index gather).
    # The step from a's column to the next is lg[..., 2] - lg[..., 1] when a is ALT, else lg[..., 1] - lg[..., 0]
    # (the same subtraction of the same operands as gathering both columns first).
    zero, one, two = lg[..., 0], lg[..., 1], lg[..., 2]
    upper, lower = two - one, one - zero
    e0 = np.where(same, two - zero, np.where(alleles_1 >= 1, upper, lower))
    valid0 = (alleles_0 >= 0) & (same | (alleles_1 >= 0))
    e1 = np.where(alleles_0 >= 1, upper, lower)
    valid1 = (alleles_1 >= 0) & (alleles_0 >= 0) & ~same
    return e0, valid0, e1, valid1


def allele_evidence(
    X: np.ndarray, site_bin: np.ndarray, alleles: np.ndarray, site_gl: np.ndarray, excluded: np.ndarray, eps: float
) -> np.ndarray:
    """float64 [S, m]: E(x, j), the summed carrier evidence for allele 1 of every state at the component's released
    sites. X int64 [N, 2, B] (the solve's MAP labels), site_bin int64 [m], alleles int8 [S, m], site_gl float32
    [N, m, 3], excluded bool [N] (kept out of the founder evidence), eps the robust mixture."""
    S, m = alleles.shape
    out = np.zeros((S, m))
    keep = ~np.asarray(excluded, dtype=bool)
    rows = np.flatnonzero(keep)
    if not len(rows) or not m:
        return out
    for lo in range(0, m, EVIDENCE_CHUNK):
        hi = min(m, lo + EVIDENCE_CHUNK)
        c = hi - lo
        bins = site_bin[lo:hi]
        lab0 = X[rows, 0][:, bins]
        lab1 = X[rows, 1][:, bins]
        cols = np.arange(c)[None, :]
        a0 = alleles[:, lo:hi][lab0, cols]
        a1 = alleles[:, lo:hi][lab1, cols]
        lg = _robust_log(site_gl[rows, lo:hi], eps)
        e0, v0, e1, v1 = _terms(lab0, lab1, a0, a1, lg)
        flat = np.concatenate(((lab0 * c + cols)[v0], (lab1 * c + cols)[v1]))
        weights = np.concatenate((e0[v0], e1[v1]))
        out[:, lo:hi] = np.bincount(flat, weights=weights, minlength=S * c).reshape(S, c)
    return out


@overload
def flip_probabilities(
    evidence: np.ndarray, alleles: np.ndarray, logit_prior: float, *, trace: Literal[False] = False
) -> np.ndarray: ...


@overload
def flip_probabilities(
    evidence: np.ndarray, alleles: np.ndarray, logit_prior: float, *, trace: Literal[True]
) -> PhaseTrace: ...


@overload
def flip_probabilities(
    evidence: np.ndarray, alleles: np.ndarray, logit_prior: float, *, trace: bool
) -> np.ndarray | PhaseTrace: ...


def flip_probabilities(
    evidence: np.ndarray, alleles: np.ndarray, logit_prior: float, *, trace: bool = False
) -> np.ndarray | PhaseTrace:
    """float64 [S, m]: capped empirical-Bayes per-copy disagreement with the selected allele [S, m] (-1: 0).
    Evidence [S, m] may have the sample's direct term removed; carrier/panel selection stays conditioned upon."""
    p_one = 1.0 / (1.0 + np.exp(-np.clip(logit_prior + evidence, -LOGIT_LIMIT, LOGIT_LIMIT)))
    flips = np.where(alleles == 1, 1.0 - p_one, p_one)
    audit = DecisionTrace(extended=True) if trace else None
    if audit is not None:
        audit.difference("cal2_cap", flips, FLIP_CEILING, flips > FLIP_CEILING)
    flips = np.minimum(flips, FLIP_CEILING)
    if audit is not None:
        audit.difference("cal2_negligible", flips, FLIP_NEGLIGIBLE, flips < FLIP_NEGLIGIBLE)
        audit.emit("cal2_unknown", alleles < 0, alleles)
    flips[(flips < FLIP_NEGLIGIBLE) | (alleles < 0)] = 0.0
    value = np.ascontiguousarray(flips)
    return PhaseTrace(value, tuple(audit.rows)) if audit is not None else value


def own_terms(
    labels: np.ndarray,
    alleles: np.ndarray,
    gl: np.ndarray,
    eps: float,
    excluded: bool,
    evidence: np.ndarray,
    logit_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One sample's own pair at the released sites: (own_state int64 [2, m] (-1: none), own_flip float64 [2, m]),
    the labels of its MAP pair (labels int64 [2, m]) whose evidence holds its reads, and their flip probabilities with
    its direct term removed, conditional on the fitted carriers and panel. A founder-excluded sample has none. ``gl``
    [m, 3] is its site likelihood,
    ``eps`` the robust mixture, ``evidence`` [S, m] the solve's E."""
    m = labels.shape[1]
    state = np.full((2, m), -1, dtype=np.int64)
    flip = np.zeros((2, m))
    if excluded or not m:
        return state, flip
    cols = np.arange(m)
    a0 = alleles[labels[0], cols]
    a1 = alleles[labels[1], cols]
    e0, v0, e1, v1 = _terms(labels[0], labels[1], a0, a1, _robust_log(gl, eps))
    for t, (lab, a, e, valid) in enumerate(((labels[0], a0, e0, v0), (labels[1], a1, e1, v1))):
        if not valid.any():
            continue
        state[t, valid] = lab[valid]
        rest = evidence[lab[valid], cols[valid]] - e[valid]
        flip[t, valid] = flip_probabilities(rest[None, :], a[valid][None, :].astype(np.int8), logit_prior)[0]
    return state, flip


__all__ = ["allele_prior", "allele_evidence", "flip_probabilities", "own_terms"]
