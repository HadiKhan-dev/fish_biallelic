"""Full-marker exclusion windows and exact pattern encodings for certified scans.
Method: docs/methods/pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts
PE1–PE3 preserve this 3.5 method.
"""

from __future__ import annotations

import numpy as np
from numba import prange

from ...contracts.pedigree import ExclusionEvidence
from ...core.kernels import kernel
from .kernels.signatures import OVERLOADS
from .parameters import EXCLUSION_DYAD_BATCH, EXCLUSION_PATTERN_LIMIT, EXCLUSION_TINY

MAX_PATTERNS = EXCLUSION_PATTERN_LIMIT

DYAD_BATCH = EXCLUSION_DYAD_BATCH


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


@kernel(overloads=OVERLOADS["exclusion._ratio"], cache=True, inline="always")
def _ratio(x0, x1, x2, y0, y1, y2, delta):
    """A / D of one jointly observed site (see the module docstring)."""
    compatible = max(x0 * max(y0, y1), x1 * max(y0, y1, y2), x2 * max(y1, y2))
    unrestricted = max(x0, x1, x2) * max(y0, y1, y2)
    bound = max(compatible, (1 - delta) * compatible + delta * unrestricted)
    return (x0 * y2 + x2 * y0) / 2 / max(bound, EXCLUSION_TINY)


@kernel(overloads=OVERLOADS["exclusion._log_bet"], cache=True, inline="always")
def _log_bet(ratio, bet):
    return np.log(max(1 - bet + bet * ratio, EXCLUSION_TINY))


@kernel(overloads=OVERLOADS["exclusion._encode_patterns"], cache=True, parallel=True)
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


def combine(evidence: list[ExclusionEvidence]) -> tuple[np.ndarray, np.ndarray]:
    """(log E [D] averaged over all configured chromosomes, jointly observed sites [D])."""
    # NumPy 2.5's reduce stub omits the ordinary allocating out=None overload.
    log_e = np.logaddexp.reduce(np.stack([value.log_e for value in evidence]), axis=0) - np.log(len(evidence))  # type: ignore[call-overload]
    observed = np.sum([value.observed_sites for value in evidence], axis=0).astype(np.int64)
    return log_e, observed
