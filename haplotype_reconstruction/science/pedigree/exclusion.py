"""Full-marker exclusion windows and exact pattern encodings for certified scans.
Method: docs/methods/pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numba import prange

from ...contracts.pedigree import ExclusionEvidence
from ...core.errors import InputError
from ...core.kernels import kernel
from .kernels.signatures import OVERLOADS
from .parameters import EXCLUSION_DYAD_BATCH, EXCLUSION_PATTERN_LIMIT, EXCLUSION_TINY

MAX_PATTERNS = EXCLUSION_PATTERN_LIMIT

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
def _ratio(x0, x1, x2, y0, y1, y2, delta):
    """A / D of one jointly observed site (see the module docstring)."""
    compatible = max(x0 * max(y0, y1), x1 * max(y0, y1, y2), x2 * max(y1, y2))
    unrestricted = max(x0, x1, x2) * max(y0, y1, y2)
    bound = max(compatible, (1 - delta) * compatible + delta * unrestricted)
    return (x0 * y2 + x2 * y0) / 2 / max(bound, EXCLUSION_TINY)


@kernel(overloads=OVERLOADS["exclusion._log_bet"], cache=True, inline="always", nogil=True)
def _log_bet(ratio, bet):
    return np.log(max(1 - bet + bet * ratio, EXCLUSION_TINY))


@kernel(overloads=OVERLOADS["exclusion._encode_patterns"], cache=True, parallel=True, nogil=True)
def _encode_patterns(gl, observed, cap):
    """Exact per-sample codes of distinct GL triples (-1 unobserved); count -1 = over the cap."""
    samples, sites, _ = gl.shape
    codes = np.empty((samples, sites), np.int16)
    catalog = np.empty((samples, cap, 3), gl.dtype)
    counts = np.zeros(samples, np.int64)
    for sample in prange(samples):
        # Establish the dictionary's key/value types without quantising floats.
        table = {(0.0, 0.0, 0.0): -1}
        del table[(0.0, 0.0, 0.0)]
        count = 0
        for site in range(sites):
            if not observed[sample, site]:
                codes[sample, site] = -1
                continue
            x0, x1, x2 = gl[sample, site]
            key = (float(x0), float(x1), float(x2))
            code = table.get(key, -1)
            if code < 0:
                if count == cap:
                    count = -1
                    break
                code = count
                table[key] = code
                catalog[sample, code] = gl[sample, site]
                count += 1
            codes[sample, site] = code
        counts[sample] = count
    return codes, catalog, counts


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
