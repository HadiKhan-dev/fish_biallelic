"""Exact repeated-likelihood reuse for full-marker Mendelian exclusion.

Integer allele depths commonly yield the same likelihood triple at many sites.
Encode those triples without rounding, then reuse each dyad's six log-bet values
for identical pattern pairs. Prefix sums still follow the original marker order;
window definitions, missingness, genotype replacement and averaging are unchanged.

The cap bounds temporary memory, not scientific resolution. A sample exceeding
it is scored through the original kernel. With 1,024 patterns per sample, the
worst-case dyad lookup occupies about 49 MiB per numerical worker; ordinary
5x inputs usually need far less. Encoding adds two bytes per sample/marker.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


MAX_PATTERNS = 1024


@njit(cache=True, parallel=True)
def encode_patterns(gl, observed, cap=MAX_PATTERNS):
    """Return exact per-sample codes/catalogs; count -1 requests direct scoring.

    Each parallel sample owns its own dictionary. Codes after a cap overflow
    are deliberately unused, since every dyad involving that sample falls back.
    Unobserved markers have code -1 and contribute the original neutral bet.
    """
    samples, sites, _ = gl.shape
    codes = np.empty((samples, sites), np.int16)
    catalog = np.empty((samples, cap, 3), gl.dtype)
    counts = np.zeros(samples, np.int64)
    for sample in prange(samples):
        # Establish the dictionary's tuple/ID types without quantizing floats.
        table = {(0., 0., 0.): -1}
        del table[(0., 0., 0.)]
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


@njit(cache=True, parallel=True)
def score_patterns(codes, catalog, counts, rows, spans, delta, bets):
    """Score dyads whose two samples were fully encoded, in original row order."""
    output = np.empty((len(rows), 2))
    for row in prange(len(rows)):
        a, b = rows[row]
        width = counts[b]
        table = np.empty((counts[a] * width, len(bets)))
        seen = np.zeros(counts[a] * width, np.bool_)
        prefix = np.empty(codes.shape[1] + 1)
        prefix[0] = 0.
        exposure = 0
        combined = -np.inf
        for bet_index in range(len(bets)):
            for site in range(codes.shape[1]):
                ca, cb = codes[a, site], codes[b, site]
                value = 0.
                if ca >= 0 and cb >= 0:
                    pair = ca * width + cb
                    if bet_index == 0:
                        exposure += 1
                    if not seen[pair]:
                        x0, x1, x2 = catalog[a, ca]
                        y0, y1, y2 = catalog[b, cb]
                        compatible = max(
                            x0 * max(y0, y1), x1 * max(y0, y1, y2), x2 * max(y1, y2))
                        unrestricted = max(x0, x1, x2) * max(y0, y1, y2)
                        bound = max(compatible, (1-delta) * compatible + delta * unrestricted)
                        ratio = (x0*y2 + x2*y0) / 2 / max(bound, 1e-300)
                        for index in range(len(bets)):
                            bet = bets[index]
                            table[pair, index] = np.log(max(1-bet + bet*ratio, 1e-300))
                        seen[pair] = True
                    value = table[pair, bet_index]
                prefix[site+1] = prefix[site] + value
            for start, stop in spans:
                combined = np.logaddexp(combined, prefix[stop] - prefix[start])
        output[row, 0] = combined - np.log(len(bets) * len(spans))
        output[row, 1] = exposure
    return output
