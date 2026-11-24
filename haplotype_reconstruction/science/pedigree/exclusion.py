"""Full-marker exclusion windows and the per-site exclusion ratio under the observation model (4.22 A11).
Method: docs/methods/pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ...contracts.pedigree import ExclusionEvidence
from ...core.errors import InputError
from ...core.kernels import kernel
from .kernels.exclusion import OVERLOADS
from .numerical_parameters import EXCLUSION_TINY
from .workspace_parameters import EXCLUSION_DYAD_BATCH

DYAD_BATCH = EXCLUSION_DYAD_BATCH


def log_threshold(sample_count: int, alpha: float) -> float:
    """Exclusion log-e threshold shared by certificates and count release."""
    return math.log(2 * sample_count / alpha)


def physical_windows(positions: np.ndarray, widths: tuple[int, ...]) -> np.ndarray:
    """int64 [W, 2] site ranges: the whole chromosome plus half-overlapping windows of each width."""
    spans = [(0, len(positions))]
    if len(positions):
        for width in widths:
            for start in range(0, int(positions[-1]) + 1, width // 2):
                a, b = np.searchsorted(positions, [start, start + width])
                if b > a:
                    spans.append((int(a), int(b)))
    return np.asarray(sorted(set(spans)), dtype=np.int64)


@kernel(overloads=OVERLOADS["exclusion._ratio"], cache=True, inline="always", nogil=True)
def _ratio(x0, x1, x2, y0, y1, y2, q0, q1, q2, ex, ey):
    """A' / D' of one site observed in both samples, under the observation model (science.shared.observation).

    Each cell's genotype likelihoods L(g) are mixed against the site's cohort genotype frequencies q at the sample's
    own outlier rate e: L'(g) = (1 - e) L(g) + e B, B = q . L (with probability e the cell's genotype is redrawn from
    q). D' = max over Mendelian-compatible genotype pairs (g, h) (no opposite homozygotes) of x'(g) y'(h) bounds the
    likelihood of the pair's cells under every parent-offspring law whose cells are redrawn independently at their
    rates; A' = (x'(0) y'(2) + x'(2) y'(0)) / 2 is the likelihood of opposite homozygotes observed through the same
    redraws, a density of the cells. Hence A' / D' has expectation at most 1 under that null at every site. No
    replacement term and no floor: the allowance is the two samples' own rates."""
    bx = q0 * x0 + q1 * x1 + q2 * x2
    by = q0 * y0 + q1 * y1 + q2 * y2
    a0 = (1.0 - ex) * x0 + ex * bx
    a1 = (1.0 - ex) * x1 + ex * bx
    a2 = (1.0 - ex) * x2 + ex * bx
    b0 = (1.0 - ey) * y0 + ey * by
    b1 = (1.0 - ey) * y1 + ey * by
    b2 = (1.0 - ey) * y2 + ey * by
    compatible = max(a0 * max(b0, b1), a1 * max(b0, b1, b2), a2 * max(b1, b2))
    return (a0 * b2 + a2 * b0) / 2 / max(compatible, EXCLUSION_TINY)


@kernel(overloads=OVERLOADS["exclusion._log_bet"], cache=True, inline="always", nogil=True)
def _log_bet(ratio, bet):
    return np.log(max(1 - bet + bet * ratio, EXCLUSION_TINY))


def contig_log_weights(configured_morgans: Sequence[float]) -> tuple[float, ...]:
    """log w_k of the included contigs in the genome e-value (DESIGN_417 D9): w_k = m_k / sum m, m_k the
    configured working-map Morgans of contig k over [0, its length] (ChromosomeSummaryView.configured_morgans).
    Fixed before any genotype is read, so sum_k w_k E_k is an e-value under any dependence between contigs
    (linearity of conditional expectation); uninformative contigs keep their weight; a contig without material
    gets -inf (no weight, no task)."""
    values = np.asarray(configured_morgans, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise InputError("exclusion weights need finite nonnegative configured Morgans for every contig")
    total = float(values.sum())
    if total <= 0:
        raise InputError("exclusion weights need configured material on some contig")
    with np.errstate(divide="ignore"):
        return tuple(float(value) for value in np.log(values / total))


def combine(evidence: Sequence[ExclusionEvidence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log endpoints of the weighted mean sum_k w_k E_k of the contigs' exclusion e-values (DESIGN_417 D9).

    Every contig with material supplies evidence carrying its fixed log weight (uninformative contigs keep their
    weight). An absent upper endpoint means exact evidence; certified lower bounds retain their supplied upper
    endpoint. Observed-site counts are summed over contigs."""
    weights = np.asarray([value.log_weight for value in evidence], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise InputError("combined exclusion evidence requires every contig's finite log weight")
    # NumPy 2.5's reduce stub omits the ordinary allocating out=None overload.
    lower = np.logaddexp.reduce(weights[:, None] + np.stack([value.log_e for value in evidence]), axis=0)
    upper = np.logaddexp.reduce(
        weights[:, None]
        + np.stack([value.log_e if value.log_e_upper is None else value.log_e_upper for value in evidence]),
        axis=0,
    )
    observed = np.sum([value.observed_sites for value in evidence], axis=0).astype(np.int64)
    return lower, upper, observed
