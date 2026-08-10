"""Enumerative code length for an unordered set of binary haplotypes.

The reversible cavity selector uses this exact combinatorial term to
regularize complete panels. It is kept in a focused module so the scientific
penalty remains independently testable without retaining obsolete alternative
K-selection controllers.
"""

from __future__ import annotations

import math


_LOG_TWO = math.log(2.0)


def _k_fits_binary_universe(n_sites: int, k: int) -> bool:
    """Return whether K is no larger than 2**L without constructing 2**L."""

    bits = k.bit_length()
    if bits <= n_sites:
        return True
    if bits > n_sites + 1:
        return False
    return (k & (k - 1)) == 0 and bits - 1 == n_sites


def log_binary_haplotype_set_count(n_sites: int, k: int) -> float:
    """Return log binomial(2**L, K) without materializing the universe.

    K is small in the supported block search. Summing K stable log-ratio
    terms avoids constructing either 2**L or the combinatorial integer.
    """

    if (
        isinstance(n_sites, bool)
        or int(n_sites) != n_sites
        or int(n_sites) < 1
    ):
        raise ValueError("n_sites must be a positive integer")
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    n_sites = int(n_sites)
    k = int(k)
    if not _k_fits_binary_universe(n_sites, k):
        raise ValueError("k cannot exceed the number of binary haplotypes")

    universe_log = n_sites * _LOG_TWO
    terms = []
    for index in range(k):
        fraction = math.ldexp(float(index), -n_sites)
        terms.append(
            universe_log
            + math.log1p(-fraction)
            - math.log(index + 1)
        )
    result = math.fsum(terms)
    if result < 0.0 and result > -1e-12:
        return 0.0
    if result < 0.0 or not math.isfinite(result):
        raise ArithmeticError("invalid enumerative haplotype-set code length")
    return result


def _selftest() -> None:
    for n_sites in range(1, 10):
        universe = 1 << n_sites
        for k in range(1, min(universe, 12) + 1):
            expected = math.log(math.comb(universe, k))
            observed = log_binary_haplotype_set_count(n_sites, k)
            if not math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=2e-12
            ):
                raise AssertionError(
                    f"binary-set code mismatch for L={n_sites}, K={k}"
                )
    try:
        log_binary_haplotype_set_count(1, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("K greater than 2**L must be rejected")


if __name__ == "__main__":
    _selftest()
    print("bhd_haplotype_mdl self-test: passed")
