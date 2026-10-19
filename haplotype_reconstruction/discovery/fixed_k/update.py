"""H step: Gauss-Seidel founder-bit updates for fixed assignments.

Founders are updated in stable decreasing order of usage (copies assigned; founders without
copies keep their alleles). For founder k at site l the cost of allele 0 and 1 is summed over
its carriers, in ascending sample order within each bucket: homozygous carriers (genotype 0 vs
2), carriers with a known partner j (genotype H[j, l] vs H[j, l] + 1, reading partners already
updated in this sweep) and carriers with an unknown copy (the row-unknown costs); every cost is
capped at the cell's unknown-unknown cost. The allele changes when the two sums differ by at
least the block's ``h_step`` tie quantum (ties keep the current allele; release 2.1, was 1e-9). Sites are independent, so the sweep runs site-parallel
(several threads) or founder-outer over site-contiguous tables (one thread) with the same terms
in the same order.
"""
from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from ..conventions import rev



@njit(cache=True, nogil=True)
def founder_buckets(assignments, k):
    """Every founder's carriers by bucket in ascending sample order, and usage (copies)."""
    n = assignments.shape[0]
    unknown = k
    homozygous = np.empty((k, n), dtype=np.int64)
    with_partner = np.empty((k, n), dtype=np.int64)
    with_unknown = np.empty((k, n), dtype=np.int64)
    partner = np.empty((k, n), dtype=np.int64)
    n_homozygous = np.zeros(k, dtype=np.int64)
    n_with_partner = np.zeros(k, dtype=np.int64)
    n_with_unknown = np.zeros(k, dtype=np.int64)
    usage = np.zeros(k, dtype=np.int64)
    for s in range(n):
        a0 = assignments[s, 0]
        a1 = assignments[s, 1]
        if a0 != unknown:
            usage[a0] += 1
        if a1 != unknown:
            usage[a1] += 1
        if a0 != unknown:
            if a0 == a1:
                homozygous[a0, n_homozygous[a0]] = s
                n_homozygous[a0] += 1
            elif a1 == unknown:
                with_unknown[a0, n_with_unknown[a0]] = s
                n_with_unknown[a0] += 1
            else:
                with_partner[a0, n_with_partner[a0]] = s
                partner[a0, n_with_partner[a0]] = a1
                n_with_partner[a0] += 1
        if a1 != unknown and a1 != a0:
            with_partner[a1, n_with_partner[a1]] = s
            partner[a1, n_with_partner[a1]] = a0
            n_with_partner[a1] += 1
    return (homozygous, with_partner, with_unknown, partner, n_homozygous, n_with_partner,
            n_with_unknown, usage)


@njit(cache=True, nogil=True)
def usage_order(usage, reverse):
    """Founders with positive usage in stable decreasing-usage order (lower index on ties; higher under reversed
    conventions)."""
    k = usage.shape[0]
    order = np.arange(k, dtype=np.int64)
    if reverse:
        order = order[::-1].copy()
    for index in range(1, k):
        current = order[index]
        position = index
        while position > 0 and usage[current] > usage[order[position - 1]]:
            order[position] = order[position - 1]
            position -= 1
        order[position] = current
    positive = 0
    while positive < k and usage[order[positive]] > 0:
        positive += 1
    return order[:positive]


@njit(cache=True, parallel=True, nogil=True)
def _sweep_site_parallel(haplotypes, order, homozygous, with_partner, with_unknown, partner,
                         n_homozygous, n_with_partner, n_with_unknown, genotype_cost,
                         unknown_cost, tie):
    """Ordered sweep, parallel over sites; costs laid out (L, N, c)."""
    length = haplotypes.shape[1]
    changes = 0
    for l in prange(length):
        for position in range(order.shape[0]):
            k = order[position]
            current = haplotypes[k, l]
            nll0 = 0.0
            nll1 = 0.0
            for index in range(n_homozygous[k]):
                s = homozygous[k, index]
                nll0 += genotype_cost[l, s, 0]
                nll1 += genotype_cost[l, s, 2]
            for index in range(n_with_partner[k]):
                s = with_partner[k, index]
                partner_allele = haplotypes[partner[k, index], l]
                nll0 += genotype_cost[l, s, partner_allele]
                nll1 += genotype_cost[l, s, partner_allele + 1]
            for index in range(n_with_unknown[k]):
                s = with_unknown[k, index]
                nll0 += unknown_cost[l, s, 0]
                nll1 += unknown_cost[l, s, 1]
            difference = nll0 - nll1
            if difference < 0.0:
                difference = -difference
            if difference < tie:
                value = current
            elif nll0 < nll1:
                value = 0
            else:
                value = 1
            haplotypes[k, l] = value
            if value != current:
                changes += 1
    return changes


@njit(cache=True, nogil=True)
def sweep_one_thread(haplotypes, order, homozygous, with_partner, with_unknown, partner,
                     n_homozygous, n_with_partner, n_with_unknown, genotype_cost_by_sample,
                     unknown_cost_by_sample, tie):
    """Ordered sweep on one thread, founder-outer and site-inner over (N, c, L) tables.

    A partner-bucket term is the cost at the partner's allele h (allele 0) or h + 1 (allele 1),
    selected on the float64 bit patterns with the mask -h, which copies the chosen cost's bits
    and keeps the site loop branch-free (vectorisable).
    """
    length = haplotypes.shape[1]
    nll0 = np.empty(length, dtype=np.float64)
    nll1 = np.empty(length, dtype=np.float64)
    cost_bits = genotype_cost_by_sample.view(np.int64)
    pick0 = np.empty(length, dtype=np.int64)
    pick1 = np.empty(length, dtype=np.int64)
    picked0 = pick0.view(np.float64)
    picked1 = pick1.view(np.float64)
    changes = 0
    for position in range(order.shape[0]):
        k = order[position]
        for l in range(length):
            nll0[l] = 0.0
            nll1[l] = 0.0
        for index in range(n_homozygous[k]):
            s = homozygous[k, index]
            cost0 = genotype_cost_by_sample[s, 0]
            cost2 = genotype_cost_by_sample[s, 2]
            for l in range(length):
                nll0[l] += cost0[l]
                nll1[l] += cost2[l]
        for index in range(n_with_partner[k]):
            s = with_partner[k, index]
            partner_alleles = haplotypes[partner[k, index]]
            bits0 = cost_bits[s, 0]
            bits1 = cost_bits[s, 1]
            bits2 = cost_bits[s, 2]
            for l in range(length):
                mask = -partner_alleles[l]
                pick0[l] = (bits1[l] & mask) | (bits0[l] & ~mask)
                pick1[l] = (bits2[l] & mask) | (bits1[l] & ~mask)
            for l in range(length):
                nll0[l] += picked0[l]
                nll1[l] += picked1[l]
        for index in range(n_with_unknown[k]):
            s = with_unknown[k, index]
            unknown0 = unknown_cost_by_sample[s, 0]
            unknown1 = unknown_cost_by_sample[s, 1]
            for l in range(length):
                nll0[l] += unknown0[l]
                nll1[l] += unknown1[l]
        alleles = haplotypes[k]
        for l in range(length):
            current = alleles[l]
            difference = nll0[l] - nll1[l]
            if difference < 0.0:
                difference = -difference
            if difference < tie:
                value = current
            elif nll0[l] < nll1[l]:
                value = 0
            else:
                value = 1
            alleles[l] = value
            if value != current:
                changes += 1
    return changes


def update_founders(workspace, haplotypes, assignments) -> int:
    """One H step in place on the int64 C-contiguous ``haplotypes``; returns the bits changed."""
    k = haplotypes.shape[0]
    (homozygous, with_partner, with_unknown, partner, n_homozygous, n_with_partner,
     n_with_unknown, usage) = founder_buckets(assignments, k)
    order = usage_order(usage, rev("d3"))
    if numba.get_num_threads() == 1:
        return int(sweep_one_thread(
            haplotypes, order, homozygous, with_partner, with_unknown, partner, n_homozygous,
            n_with_partner, n_with_unknown, workspace.genotype_cost_by_sample,
            workspace.unknown_cost_by_sample, float(workspace.quanta.h_step)))
    return int(_sweep_site_parallel(
        haplotypes, order, homozygous, with_partner, with_unknown, partner, n_homozygous,
        n_with_partner, n_with_unknown, workspace.genotype_cost_by_site,
        workspace.unknown_cost_by_site, float(workspace.quanta.h_step)))
