"""Integer sufficient counts for parental copy factors; no floating running sums.

Each cache belongs to one mutable state (or one immutable release state). Slots
are counted by selected parental homologue and child's allele. Queries remove
both slots of the excluded group, including a repeated parent if present.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def refresh(cache, genotype, selector, offsets, groups, slots, decoded):
    index, counts, parents, children = cache
    for parent in prange(len(index)):
        row = index[parent]
        if row < 0:
            continue
        counts[row, :, :, :] = 0
        for k in range(offsets[parent], offsets[parent + 1]):
            group, slot = groups[k], slots[k]
            if decoded is not None and not decoded[group]:
                continue
            child = children[group]
            for site in range(genotype.shape[1]):
                allele = (genotype[child, site] >> (1 - slot)) & 1
                homologue = (selector[group, site] >> (1 - slot)) & 1
                counts[row, site, homologue, allele] += 1


@njit(cache=True, nogil=True)
def add_terms(cache, parent, site, exclude, genotype, selector, log_keep, log_error, out):
    index, counts, parents, children = cache
    row = index[parent]
    # Four scalar sufficient counts avoid a heap allocation in every site query.
    n00 = counts[row, site, 0, 0]
    n01 = counts[row, site, 0, 1]
    n10 = counts[row, site, 1, 0]
    n11 = counts[row, site, 1, 1]
    if exclude >= 0:
        for slot in range(2):
            if parents[exclude, slot] == parent:
                allele = (genotype[children[exclude], site] >> (1 - slot)) & 1
                homologue = (selector[exclude, site] >> (1 - slot)) & 1
                if homologue == 0:
                    if allele == 0:
                        n00 -= 1
                    else:
                        n01 -= 1
                else:
                    if allele == 0:
                        n10 -= 1
                    else:
                        n11 -= 1
    total = n00 + n01 + n10 + n11
    for g in range(4):
        matches = (n00 if (g >> 1) == 0 else n01) + (n10 if (g & 1) == 0 else n11)
        out[g] += matches * log_keep + (total - matches) * log_error



@njit(cache=True, nogil=True)
def update_group(cache, group, lo, old_genotype, old_selector, genotype, selector, add_only=False):
    index, counts, parents, children = cache
    child = children[group]
    for j in range(len(old_genotype)):
        site = lo + j
        old_g, old_s = old_genotype[j], old_selector[j]
        new_g, new_s = genotype[child, site], selector[group, site]
        if not add_only and old_g == new_g and old_s == new_s:
            continue
        for slot in range(2):
            parent = parents[group, slot]
            if parent < 0 or index[parent] < 0:
                continue
            row = index[parent]
            if not add_only:
                counts[row, site, (old_s >> (1 - slot)) & 1, (old_g >> (1 - slot)) & 1] -= 1
            counts[row, site, (new_s >> (1 - slot)) & 1, (new_g >> (1 - slot)) & 1] += 1


def build(genotype, selector, graph, offsets, groups, slots, decoded=None):
    # Degree-one factors cannot reuse a sibling query; keep their original fold.
    parents = np.flatnonzero(np.diff(offsets) >= 2)
    index = np.full(len(offsets) - 1, -1, dtype=np.int64)
    index[parents] = np.arange(len(parents))
    cache = (index, np.zeros((len(parents), genotype.shape[1], 2, 2), np.int32),
             graph.group_parent, graph.group_child)
    refresh(cache, genotype, selector, offsets, groups, slots, decoded)
    return cache
