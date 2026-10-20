"""H step: Gauss-Seidel founder-bit updates for fixed assignments.

Founders are updated in stable decreasing order of expected-copy usage (founders without
copies keep their alleles). Tied samples share each existing state cost uniformly over their
winning category. Homozygous states count once in evidence and twice in usage. For founder k at site l the cost of allele 0 and 1 is summed over
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
from ..residual_births import _accurate_sum



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


@njit(cache=True, nogil=True)
def _fractional_usage_commutes(usage, row_ptr, partner_ptr, partners):
    """No tied-usage founders share a heterozygote term, so their updates commute."""
    for founder in range(len(usage)):
        if usage[founder] <= 0:
            continue
        for entry in range(row_ptr[founder], row_ptr[founder + 1]):
            for at in range(partner_ptr[entry], partner_ptr[entry + 1]):
                if usage[founder] == usage[partners[at]]:
                    return False
    return True


@njit(cache=True, nogil=True)
def _singleton_usage_commutes(usage, partner, n_with_partner):
    for founder in range(len(usage)):
        if usage[founder] <= 0:
            continue
        for entry in range(n_with_partner[founder]):
            if usage[founder] == usage[partner[founder, entry]]:
                return False
    return True


def update_founders(workspace, haplotypes, assignments, tie_mask=None, *,
                    schedule_reverse=None, with_usage_tie=False, with_transition_certificate=False):
    """One H sweep; an explicit schedule reverses only exact positive-usage ties."""
    k = haplotypes.shape[0]
    reverse = rev("d3") if schedule_reverse is None else bool(schedule_reverse)
    if tie_mask is None:
        from .assign import tie_mask as membership
        tie_mask = membership(workspace, haplotypes)
    fractional = np.any(np.count_nonzero(tie_mask, axis=1) > 1)
    if fractional:
        buckets = fractional_buckets(tie_mask, k)
        usage = buckets[-1]
        order = usage_order(usage, reverse)
        sweep = fractional_sweep_one_thread if numba.get_num_threads() == 1 else _fractional_sweep_parallel
        changes = int(sweep(haplotypes, order, *buckets[:-1], workspace.genotype_cost_by_sample,
                            workspace.unknown_cost_by_sample, float(workspace.quanta.h_step)))
    else:
        (homozygous, with_partner, with_unknown, partner, n_homozygous, n_with_partner,
         n_with_unknown, usage) = founder_buckets(assignments, k)
        order = usage_order(usage, reverse)
        if numba.get_num_threads() == 1:
            changes = int(sweep_one_thread(
                haplotypes, order, homozygous, with_partner, with_unknown, partner, n_homozygous,
                n_with_partner, n_with_unknown, workspace.genotype_cost_by_sample,
                workspace.unknown_cost_by_sample, float(workspace.quanta.h_step)))
        else:
            changes = int(_sweep_site_parallel(
                haplotypes, order, homozygous, with_partner, with_unknown, partner, n_homozygous,
                n_with_partner, n_with_unknown, workspace.genotype_cost_by_site,
                workspace.unknown_cost_by_site, float(workspace.quanta.h_step)))
    tied = bool(np.any(usage[order[1:]] == usage[order[:-1]]))
    if with_transition_certificate:
        commuting = (not tied or
                     (_fractional_usage_commutes(usage, buckets[0], buckets[5], buckets[6])
                      if fractional else _singleton_usage_commutes(usage, partner, n_with_partner)))
        return changes, tied, bool(commuting)
    return (changes, tied) if with_usage_tie else changes


@njit(cache=True, nogil=True)
def fractional_buckets(mask, k):
    """Sparse founder/sample entries and integer state counts, with expected-copy usage.

    At a site only the partner-allele counts change with Gauss-Seidel updates. One
    homozygous state is one cost term but contributes two copies to usage.
    """
    n = mask.shape[0]
    n_pairs = k * (k + 1) // 2
    lookup = np.empty((k, k), dtype=np.int64)
    state = 0
    for a in range(k):
        for b in range(a, k):
            lookup[a, b] = lookup[b, a] = state
            state += 1
    sizes = np.zeros(n, dtype=np.int64)
    partner_count = 0
    for s in range(n):
        for st in range(mask.shape[1]):
            sizes[s] += mask[s, st]
        for a in range(k):
            for b in range(a + 1, k):
                partner_count += 2 * int(mask[s, lookup[a, b]])
    row_ptr = np.zeros(k + 1, dtype=np.int64)
    samples = np.empty(k * n, dtype=np.int64)
    homozygous = np.empty(k * n, dtype=np.uint8)
    unknown = np.empty(k * n, dtype=np.uint8)
    denominator = np.empty(k * n, dtype=np.int64)
    partner_ptr = np.empty(k * n + 1, dtype=np.int64)
    partners = np.empty(partner_count, dtype=np.int64)
    usage = np.zeros(k, dtype=np.float64)
    entry = 0
    at = 0
    terms = np.empty(n, dtype=np.float64)
    for founder in range(k):
        row_ptr[founder] = entry
        used = 0
        for s in range(n):
            first = at
            for other in range(k):
                if other != founder and mask[s, lookup[founder, other]]:
                    partners[at] = other
                    at += 1
            h = mask[s, lookup[founder, founder]]
            u = mask[s, n_pairs + founder]
            if at != first or h or u:
                samples[entry] = s
                homozygous[entry] = h
                unknown[entry] = u
                denominator[entry] = sizes[s]
                partner_ptr[entry] = first
                terms[used] = (2 * int(h) + int(u) + at - first) / sizes[s]
                used += 1
                entry += 1
        usage[founder] = _accurate_sum(terms[:used])
    row_ptr[k] = entry
    partner_ptr[entry] = at
    return (row_ptr, samples[:entry], homozygous[:entry], unknown[:entry], denominator[:entry],
            partner_ptr[:entry + 1], partners, usage)


@njit(cache=True, nogil=True)
def _fractional_site(haplotypes, site, order, row_ptr, samples, homozygous, unknown,
                     denominator, partner_ptr, partners, genotype_cost, unknown_cost, tie):
    terms0 = np.empty(genotype_cost.shape[0], dtype=np.float64)
    terms1 = np.empty(genotype_cost.shape[0], dtype=np.float64)
    partials = np.empty(genotype_cost.shape[0], dtype=np.float64)
    changes = 0
    for position in range(len(order)):
        founder = order[position]
        start, stop = row_ptr[founder], row_ptr[founder + 1]
        for entry in range(start, stop):
            s = samples[entry]
            count0 = 0
            count1 = 0
            for at in range(partner_ptr[entry], partner_ptr[entry + 1]):
                if haplotypes[partners[at], site] == 0:
                    count0 += 1
                else:
                    count1 += 1
            # Aggregate integer counts before weighting, so ordering tied partners cannot
            # affect a sample's contribution. Read partners updated earlier in this sweep.
            h, u = int(homozygous[entry]), int(unknown[entry])
            terms0[entry - start] = ((h + count0) * genotype_cost[s, 0, site]
                                    + count1 * genotype_cost[s, 1, site]
                                    + u * unknown_cost[s, 0, site]) / denominator[entry]
            terms1[entry - start] = (count0 * genotype_cost[s, 1, site]
                                    + (h + count1) * genotype_cost[s, 2, site]
                                    + u * unknown_cost[s, 1, site]) / denominator[entry]
        nll0 = _accurate_sum(terms0[:stop - start], partials)
        nll1 = _accurate_sum(terms1[:stop - start], partials)
        current = haplotypes[founder, site]
        value = current
        if abs(nll0 - nll1) >= tie:
            value = 0 if nll0 < nll1 else 1
        haplotypes[founder, site] = value
        changes += value != current
    return changes


@njit(cache=True, nogil=True)
def fractional_sweep_one_thread(haplotypes, order, row_ptr, samples, homozygous, unknown,
                                 denominator, partner_ptr, partners, genotype_cost, unknown_cost, tie):
    # Each site's Gauss-Seidel order is unchanged. Moving independent sites inside
    # the sample loop reads the cost and founder rows contiguously, and reuses the
    # expansion-sum scratch instead of allocating it for every founder/site pair.
    n_samples, _, length = genotype_cost.shape
    terms0 = np.empty((n_samples, length), dtype=np.float64)
    terms1 = np.empty((n_samples, length), dtype=np.float64)
    partials = np.empty(n_samples, dtype=np.float64)
    counts1 = np.empty(length, dtype=np.int64)
    changes = 0
    for position in range(len(order)):
        founder = order[position]
        start, stop = row_ptr[founder], row_ptr[founder + 1]
        for entry in range(start, stop):
            sample = samples[entry]
            for site in range(length):
                counts1[site] = 0
            begin, end = partner_ptr[entry], partner_ptr[entry + 1]
            for at in range(begin, end):
                partner = partners[at]
                for site in range(length):
                    counts1[site] += haplotypes[partner, site]
            h, u = int(homozygous[entry]), int(unknown[entry])
            for site in range(length):
                count1 = counts1[site]
                count0 = end - begin - count1
                terms0[entry - start, site] = ((h + count0) * genotype_cost[sample, 0, site]
                                              + count1 * genotype_cost[sample, 1, site]
                                              + u * unknown_cost[sample, 0, site]) / denominator[entry]
                terms1[entry - start, site] = (count0 * genotype_cost[sample, 1, site]
                                              + (h + count1) * genotype_cost[sample, 2, site]
                                              + u * unknown_cost[sample, 1, site]) / denominator[entry]
        for site in range(length):
            nll0 = _accurate_sum(terms0[:stop - start, site], partials)
            nll1 = _accurate_sum(terms1[:stop - start, site], partials)
            current = haplotypes[founder, site]
            value = current
            if abs(nll0 - nll1) >= tie:
                value = 0 if nll0 < nll1 else 1
            haplotypes[founder, site] = value
            changes += value != current
    return changes


@njit(cache=True, parallel=True, nogil=True)
def _fractional_sweep_parallel(haplotypes, order, row_ptr, samples, homozygous, unknown,
                               denominator, partner_ptr, partners, genotype_cost, unknown_cost, tie):
    changes = 0
    for site in prange(haplotypes.shape[1]):
        changes += _fractional_site(haplotypes, site, order, row_ptr, samples, homozygous, unknown,
                                    denominator, partner_ptr, partners, genotype_cost, unknown_cost, tie)
    return changes
