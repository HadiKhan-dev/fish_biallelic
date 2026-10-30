"""Release of one component: calls, phase, phase sets and their uncertainty from exact block posteriors.

For each sample, every other sample fixed at the converged MAP, the block's exact forward-backward
(kernels.forward_backward) gives posteriors of its ordered label pair per bin. The site model turns them into per-site
ordered allele posteriors (kernels.site_posteriors): given its label pair, a sample carries the labels' panel alleles
except for a departure with probability ``site_deviation``, and its own reads at the site weigh the alternatives.

- Genotypes come from the unanchored posterior (the homologue-order gauge does not change unordered genotypes). A
  genotype is called where its posterior reaches ``genotype_posterior``; otherwise both alleles are unknown.
- Phase of a sample with a linked parent is anchored by parental origin (homologue k is the slot assigned to the
  slot-k parent: its copy where the chain's source is 0/1).
  A sample without a linked parent but with children is anchored by its children's chains. A sample with neither is a
  gauge sample: its chromosome is partitioned at hard cuts (forced breaks, below) and at every heterozygous bin whose
  ordered posterior, given the current partition's anchor, falls below ``phase_posterior``. Each partition is anchored
  at its first confidently heterozygous bin (P(heterozygous) >= ``phase_posterior``) by the gauge "homologue 0
  carries the smaller label" (no decoded label is assumed); a partition without such a bin releases no phase.
- Forced breaks (``break_bins``, chosen by the stage): a sample without a linked parent breaks its phase sets at its
  painting tie boundaries (the painting's track order there is a convention), and every sample breaks at the junction
  candidates where it carries both junction labels, unless the parental homologue it copies there is pinned by the
  parent's own reads (model.withheld_joins). A break is recorded at its own bin, never moved to the next anchor,
  and no phase is released across it.
- A heterozygous call is released phased where its ordered posterior reaches ``phase_posterior``; otherwise it is
  released unphased (frame_undetermined bits 0 and 1).
"""
from __future__ import annotations

import numpy as np

from . import kernels


def label_pair_posterior(post, labs0, labs1, S):
    """float64 [B, S, S] ordered label-pair posteriors from block-state posteriors post [B, n0, n1]."""
    B = post.shape[0]
    out = np.zeros((B, S, S))
    for b in range(B):
        l0, l1 = labs0[b], labs1[b]
        np.add.at(out[b], (l0[:, None].repeat(len(l1), 1), l1[None, :].repeat(len(l0), 0)), post[b])
    return out


def release_sample(model, n, ctx):
    """One sample's release over the component's released sites.

    ctx: site_bin int64 [m] (bin of each site), site_alleles int8 [S, m] (painting-state alleles at the sites),
    site_gl float32 [m, 3] (this sample's genotype likelihoods at the sites), break_bins (this sample's painting tie
    boundaries and withheld junction joins, as bins), config.
    Returns (calls int8 [m, 2], undetermined bool [m], set_breaks: sorted bins b > 0 where a new phase set starts
    (the phase across the edge between bins b - 1 and b is not released)). With ctx["scores"] or ctx["chain"] a
    fourth element, a dict, adds "scores": float32 [2, m], the released genotype's posterior and the ordered allele
    posterior max(P01, P10) of each site (conditional scores, for reporting), and "chain": float32 [2, 2, B], for
    each linked homologue k the meiosis chain's P(source 0) and P(source 1) per bin (the unanchored block posterior;
    NaN for a free homologue).
    """
    config = ctx["config"]
    tau_g, tau_p = config.genotype_posterior, config.phase_posterior
    B, S = model.B, model.S
    (em, T0, T1, TG, labs0, labs1), kinds = model.tables(n)
    anchored_by_parent = bool(np.any(kinds == 1))
    children = len(model.child_tracks[n])
    gauge = not anchored_by_parent and children == 0
    forced = ctx["break_bins"]
    every = np.ones((1, 1), dtype=np.bool_)
    post = kernels.forward_backward(em, T0, T1, TG, -1, every)
    gposts = label_pair_posterior(post, labs0, labs1, S)
    map0, map1 = model.X[n, 0], model.X[n, 1]
    het_bins = np.flatnonzero(map0 != map1)
    cuts = sorted({int(b) for b in forced if 0 < b < B})
    set_breaks = []
    if gauge and len(het_bins):
        posts = np.zeros((B, S, S))                             # bins of a partition without an anchor: no phase
        het_conf = 1.0 - np.einsum("bxx->b", gposts)            # P(the label pair is heterozygous)
        candidates = [int(b) for b in het_bins if het_conf[b] >= tau_p]
        start = 0
        while start < B:
            if start > 0:
                set_breaks.append(start)
            hard = next((c for c in cuts if c > start), B)
            anchor = next((b for b in candidates if start <= b < hard), None)
            if anchor is None:
                start = hard
                continue
            mask = labs0[anchor][:, None] < labs1[anchor][None, :]
            lp = label_pair_posterior(kernels.forward_backward(em, T0, T1, TG, anchor, mask), labs0, labs1, S)
            m0, m1 = (map0, map1) if map0[anchor] < map1[anchor] else (map1, map0)
            conf = lp[np.arange(B), m0, m1]                    # absolute posterior of the decoded ordered pair
            stop = next((int(b) for b in het_bins
                         if anchor < b < hard and het_conf[b] >= tau_g and conf[b] < tau_p), hard)
            posts[start:stop] = lp[start:stop]
            start = stop
    else:
        posts = gposts
        set_breaks = cuts
    site_bin = ctx["site_bin"]
    alleles = ctx["site_alleles"]
    gl = ctx["site_gl"]
    ordered = kernels.site_posteriors(posts, site_bin, alleles, gl, config.site_deviation)
    genotype = kernels.site_posteriors(gposts, site_bin, alleles, gl, config.site_deviation)
    pg = np.stack([genotype[0], genotype[1] + genotype[2], genotype[3]])
    g = pg.argmax(axis=0)
    called = pg.max(axis=0) >= tau_g
    p01, p10 = ordered[1], ordered[2]
    het = called & (g == 1)
    phased = het & (np.maximum(p01, p10) >= tau_p)
    calls = np.full((len(site_bin), 2), -1, dtype=np.int8)
    hom = called & (g != 1)
    calls[hom, 0] = calls[hom, 1] = (g[hom] // 2).astype(np.int8)
    one_first = p10 > p01                                       # allele 1 on homologue 0
    calls[het, 0] = np.where(one_first[het], 1, 0)
    calls[het, 1] = np.where(one_first[het], 0, 1)
    extras = {}
    if ctx.get("scores"):
        extras["scores"] = np.stack([pg.max(axis=0), np.maximum(p01, p10)]).astype(np.float32)
    if ctx.get("chain"):
        chain = np.full((2, 2, B), np.nan, dtype=np.float32)
        for k in (0, 1):
            if kinds[k] == 1:
                source = post.sum(axis=2 - k)                  # [B, n_k]: homologue k's state marginal
                chain[k, 0] = source[:, 0]
                chain[k, 1] = source[:, 1]
        extras["chain"] = chain
    if extras:
        return calls, het & ~phased, set_breaks, extras
    return calls, het & ~phased, set_breaks
