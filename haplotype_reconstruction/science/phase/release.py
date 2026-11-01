"""Release of one component: calls, phase, phase sets and their uncertainty from exact block posteriors.

For each sample, every other sample fixed at the converged MAP, the block's exact forward-backward
(kernels.forward_backward) gives posteriors of its ordered label pair per bin. The site model turns them into per-site
ordered allele posteriors (kernels.site_posteriors): given its label pair, each homologue carries its label's panel
allele except for a flip with that allele's leave-one-out uncertainty (allele_evidence.py: the probability that the
state's allele is the other one given every other carrier's reads, so the sample's own reads are not counted twice),
the pair departs from the alleles with probability ``site_deviation``, and its own reads at the site weigh the
alternatives.

- Genotypes come from the unanchored posterior (the homologue-order gauge does not change unordered genotypes). A
  genotype is called where its posterior reaches ``genotype_posterior``; otherwise both alleles are unknown.
- Phase of a sample with a linked parent is anchored by parental origin (homologue k is the slot assigned to the
  slot-k parent: its copy where the chain's source is 0/1).
  A sample without a linked parent but with children is anchored by its children's chains. A sample with neither is a
  gauge sample: its chromosome is partitioned at hard cuts (forced breaks, below) and at every heterozygous bin where,
  given the current partition's anchor, the orientation of its alleles is uncertain: the median over the bin's
  confidently heterozygous sites (P(heterozygous) >= ``genotype_posterior``) of max(P01, P10) / (P01 + P10) falls
  below ``set_posterior`` (stricter than ``phase_posterior``: a wrong continuation misphases the rest of the set).
  Uncertainty about which labels a sample carries does not break its phase where the
  candidate labels carry the same alleles at its sites (the orientation is then still determined). Each partition is
  anchored at its first confidently heterozygous bin (P(heterozygous) >= ``phase_posterior``) by the gauge "homologue
  0 carries the smaller label" (no decoded label is assumed); a partition without such a bin releases no phase.
- Forced breaks (``break_bins``, chosen by the stage): a sample without a linked parent breaks its phase sets at its
  painting tie boundaries (the painting's track order there is a convention), and every sample breaks at the junction
  candidates where it carries both junction labels, unless the parental homologue it copies there is pinned by the
  parent's own reads (model.withheld_joins). A break is recorded at its own bin, never moved to the next anchor,
  and no phase is released across it. A sample anchored by its children takes its orientation from their chains on
  both sides of a painting tie, so it keeps a tie break (one that is not also a withheld join) only where that
  orientation is uncertain beside it: where the orientation certainty above, under its children-anchored posterior,
  is below ``set_posterior`` at the nearest bin with a confidently heterozygous site on either side
  (``_bridged_ties``). Each solve decides for itself, and the agreement keeps a break that either solve keeps.
- A heterozygous call is released phased where its ordered posterior reaches ``phase_posterior``; otherwise it is
  released unphased (frame_undetermined bits 0 and 1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...contracts.phase import ReleaseTables, SampleRelease
from ...contracts.roles import record_field
from . import release_kernels as kernels
from .allele_evidence import own_terms
from .diagnostics import DecisionTrace, PhaseTrace
from .model import LabelModel
from .parameters import PhaseParameters
from .window import site_windows


@dataclass(frozen=True)
class ReleaseInputs:
    """One sample's site release inputs (P0 FP14-FP19), borrowed without mutation.

    Site axes contain only released component sites. GLs are float32[L,3] as in
    the recorded 3.5 release; all uncertainty and decision arithmetic is float64.
    """

    config: PhaseParameters = record_field("state", units="1", dtype="PhaseParameters", stored=False)
    site_bin: np.ndarray = record_field("state", units="1", dtype="int64", stored=False)
    site_alleles: np.ndarray = record_field("state", units="1", dtype="int8", stored=False)
    site_gl: np.ndarray = record_field("evidence", units="1", dtype="float32", stored=False)
    break_bins: tuple[int, ...] = record_field("call", units="1", dtype="bin_indices", stored=False)
    tie_bins: tuple[int, ...] = record_field("call", units="1", dtype="bin_indices", stored=False)
    allele_evidence: np.ndarray = record_field("state", units="nats", dtype="float64", stored=False)
    flips: np.ndarray = record_field("state", units="probability", dtype="float64", stored=False)
    logit_prior: float = record_field("state", units="nats", dtype="float64", stored=False)
    robust_epsilon: float = record_field("state", units="probability", dtype="float64", stored=False)
    excluded: bool = record_field("call", units="1", dtype="bool", stored=False)
    windowed: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=True)
    scores: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=False)
    chain: bool = record_field("diagnostic", units="1", dtype="bool", stored=False, default=True)
    audit: DecisionTrace | None = record_field(
        "diagnostic", units="1", dtype="DecisionTrace | None", stored=False, default=None
    )


def release_tables(model: LabelModel, n: int) -> ReleaseTables:
    """Build owned exact-posterior factors with private descendant scratch per sample."""
    kinds = model.kinds(n)
    values = kernels.block_tables(
        n,
        kinds,
        model.parents(n),
        model.E,
        model.X,
        model.V,
        model.par,
        model.release_desc[n],
        model.child_tracks[n],
        np.full_like(model.mark, -1),
        model.S,
        model.lstay,
        model.lmove,
        model.mstay,
        model.mmove,
        model.ain,
        model.astay,
        model.bout,
        model.init_copy,
        model.init_bottom,
    )
    emissions, transition0, transition1, gauge_transition, labels0, labels1 = values
    return ReleaseTables(emissions, transition0, transition1, gauge_transition, labels0, labels1, kinds)


def label_pair_posterior(post, labs0, labs1, S):
    """float64 [B, S, S] ordered label-pair posteriors from block-state posteriors post [B, n0, n1]."""
    B = post.shape[0]
    out = np.zeros((B, S, S))
    for b in range(B):
        l0, l1 = labs0[b], labs1[b]
        np.add.at(out[b], (l0[:, None].repeat(len(l1), 1), l1[None, :].repeat(len(l0), 0)), post[b])
    return out


def _bin_orientation(ordered, site_bin, tau_g, B, *, trace=False, audit=None):
    """float64 [B]: per bin, the median over its sites (``site_bin``) with P(heterozygous) >= tau_g under the ordered
    allele posteriors ``ordered`` (kernels.site_posteriors) of the orientation certainty max(P01, P10) / (P01 + P10);
    NaN in a bin without such a site (no evidence that the orientation is lost)."""
    out = np.full(B, np.nan)
    het = ordered[1] + ordered[2]
    selected = het >= tau_g
    if audit is not None:
        audit.difference("orientation_het", het, tau_g, selected)
    sites = np.flatnonzero(selected)
    if not len(sites):
        return (
            dict(
                values=out,
                P01=ordered[1].copy(),
                P10=ordered[2].copy(),
                cutoff=np.full(len(het), tau_g),
                selected=selected,
            )
            if trace
            else out
        )
    certainty = np.maximum(ordered[1], ordered[2])[sites] / het[sites]
    bins = site_bin[sites]
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    for a, e in zip(starts, np.r_[starts[1:], len(bins)], strict=False):
        out[bins[a]] = np.median(certainty[a:e])
    return (
        dict(
            values=out, P01=ordered[1].copy(), P10=ordered[2].copy(), cutoff=np.full(len(het), tau_g), selected=selected
        )
        if trace
        else out
    )


def _site_posteriors(post, site_bin, alleles, gl, epsilon, flips, lo=0, hi=None):
    """kernels.site_posteriors over the sites [lo, hi) with the sample's flip arrays ``flips`` (flips [S, m],
    own_state [2, m], own_flip [2, m])."""
    hi = len(site_bin) if hi is None else hi
    if lo == 0 and hi == len(site_bin):
        return kernels.site_posteriors(post, site_bin, alleles, gl, epsilon, *flips)
    return kernels.site_posteriors(
        post,
        site_bin[lo:hi],
        np.ascontiguousarray(alleles[:, lo:hi]),
        gl[lo:hi],
        epsilon,
        *(np.ascontiguousarray(a[:, lo:hi]) for a in flips),
    )


def sample_flips(model, n, ctx):
    """(flips [S, m], own_state [2, m], own_flip [2, m]) of sample n (kernels.site_posteriors): the solve's flip
    probabilities, with the labels of n's own MAP pair recomputed without n's reads (allele_evidence.own_terms)."""
    site_bin, alleles = ctx.site_bin, ctx.site_alleles
    labels = np.ascontiguousarray(model.X[n][:, site_bin])
    state, flip = own_terms(
        labels, alleles, ctx.site_gl, ctx.robust_epsilon, bool(ctx.excluded), ctx.allele_evidence, ctx.logit_prior
    )
    return ctx.flips, state, flip


def _orientation_certainty(
    post, site_bin, alleles, gl, epsilon, flips, tau_g, lo_bin, hi_bin, B, windowed=True, *, audit=None
):
    """_bin_orientation of the bins in [lo_bin, hi_bin) under the label-pair posteriors ``post``; NaN elsewhere."""
    lo = int(np.searchsorted(site_bin, lo_bin, side="left"))
    hi = int(np.searchsorted(site_bin, hi_bin, side="left"))
    if hi <= lo:
        return np.full(B, np.nan)
    if not windowed:
        ordered = _site_posteriors(post, site_bin, alleles, gl, epsilon, flips, lo, hi)
        return _bin_orientation(ordered, site_bin[lo:hi], tau_g, B, audit=audit)
    orientation = np.full(B, np.nan)
    for left, right in site_windows(site_bin, lo, hi):
        bins = site_bin[left:right]
        ordered = _site_posteriors(post, site_bin, alleles, gl, epsilon, flips, left, right)
        local = _bin_orientation(ordered, bins, tau_g, B, audit=audit)
        orientation[bins[0] : bins[-1] + 1] = local[bins[0] : bins[-1] + 1]
    return orientation


def _bridged_ties(ties, orientation, tau_s, *, audit=None):
    """The painting-tie cuts (bins) that a sample anchored by its children bridges: those where the orientation
    certainty (``orientation``, _bin_orientation under its children-anchored posterior) at the nearest bin with a
    confidently heterozygous site on each side (before the cut's bin, and at or after it) reaches tau_s. A cut without
    such a bin on one side is kept."""
    informative = np.flatnonzero(~np.isnan(orientation))
    out = set()
    for c in ties:
        k = int(np.searchsorted(informative, c, side="left"))
        if 0 < k < len(informative):
            value = min(orientation[informative[k - 1]], orientation[informative[k]])
            bridged = value >= tau_s
            if audit is not None:
                audit.difference("bridge", value, tau_s, bridged)
            if bridged:
                out.add(int(c))
    return out


def release_sample(model: LabelModel, n: int, ctx: ReleaseInputs, *, trace: bool = False) -> SampleRelease | PhaseTrace:
    """Release one sample; trace observes this same producer with private ordered operands."""
    if trace:
        audit = DecisionTrace(extended=True)
        value = _release_sample(model, n, replace(ctx, audit=audit))
        return PhaseTrace(value, tuple(audit.rows))
    return _release_sample(model, n, ctx)


def _release_sample(model: LabelModel, n: int, ctx: ReleaseInputs) -> SampleRelease:
    """One sample's release over the component's released sites.

    ctx: site_bin int64 [m] (bin of each site), site_alleles int8 [S, m] (painting-state alleles at the sites),
    site_gl float32 [m, 3] (this sample's genotype likelihoods at the sites), break_bins (this sample's painting tie
    boundaries and withheld junction joins, as bins), tie_bins (optional: the painting tie boundaries among break_bins
    that are not withheld junction joins), config; the solve's allele uncertainty (allele_evidence.py):
    allele_evidence float64 [S, m] (E), flips float64 [S, m] (d of every state from E), logit_prior, robust_epsilon,
    and excluded (this sample is kept out of the founder evidence).
    Returns a SampleRelease with calls int8[m,2], undetermined bool[m], sorted
    set-break bins, optional float32[2,m] reporting scores and optional
    float64[2,2,B] source probabilities. Free-homologue chains are NaN. Inputs
    are borrowed; arrays in the returned record are owned by this sample call.
    """
    config = ctx.config
    audit = ctx.audit
    windowed = ctx.windowed
    tau_g, tau_p, tau_s = config.genotype_posterior, config.phase_posterior, config.set_posterior
    B, S = model.B, model.S
    tables = release_tables(model, n)
    em, T0, T1, TG = tables.emissions, tables.transition0, tables.transition1, tables.gauge_transition
    labs0, labs1, kinds = tables.labels0, tables.labels1, tables.kinds
    anchored_by_parent = bool(np.any(kinds == 1))
    children = len(model.child_tracks[n])
    gauge = not anchored_by_parent and children == 0
    forced = ctx.break_bins
    site_bin = ctx.site_bin
    alleles = ctx.site_alleles
    gl = ctx.site_gl
    flips = sample_flips(model, n, ctx)
    every = np.ones((1, 1), dtype=np.bool_)
    post = kernels.forward_backward(em, T0, T1, TG, -1, every)
    gposts = label_pair_posterior(post, labs0, labs1, S)
    map0, map1 = model.X[n, 0], model.X[n, 1]
    het_bins = np.flatnonzero(map0 != map1)
    cuts = sorted({int(b) for b in forced if 0 < b < B})
    set_breaks = []
    if gauge and len(het_bins):
        posts = np.zeros((B, S, S))  # bins of a partition without an anchor: no phase
        het_conf = 1.0 - np.einsum("bxx->b", gposts)  # P(the label pair is heterozygous)
        candidates = [int(b) for b in het_bins if het_conf[b] >= tau_p]
        if audit is not None:
            audit.difference("anchor", het_conf[het_bins], tau_p, het_conf[het_bins] >= tau_p)
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
            orientation = _orientation_certainty(
                lp, site_bin, alleles, gl, config.site_deviation, flips, tau_g, anchor, hard, B, windowed, audit=audit
            )
            stop = next(
                (int(b) for b in het_bins if anchor < b < hard and het_conf[b] >= tau_g and orientation[b] < tau_s),
                hard,
            )
            if audit is not None:
                for b in het_bins:
                    if anchor < b < hard and b <= stop:
                        included = het_conf[b] >= tau_g
                        audit.difference("stop_het", het_conf[b], tau_g, included)
                        if included:
                            audit.difference("stop_orientation", orientation[b], tau_s, orientation[b] < tau_s)
            posts[start:stop] = lp[start:stop]
            start = stop
    else:
        posts = gposts
        set_breaks = cuts
    ordered = None if windowed else _site_posteriors(posts, site_bin, alleles, gl, config.site_deviation, flips)
    ties = ctx.tie_bins
    if children and not anchored_by_parent and len(ties):
        orientation = (
            _orientation_certainty(
                posts, site_bin, alleles, gl, config.site_deviation, flips, tau_g, 0, B, B, audit=audit
            )
            if windowed
            else _bin_orientation(ordered, site_bin, tau_g, B, audit=audit)
        )
        bridged = _bridged_ties(ties, orientation, tau_s, audit=audit)
        set_breaks = [c for c in cuts if c not in bridged]
    # Only a single sample's result survives these site windows. The stage consumes completed samples immediately.
    calls = np.full((len(site_bin), 2), -1, dtype=np.int8)
    undetermined = np.zeros(len(site_bin), dtype=bool)
    scores = np.empty((2, len(site_bin)), dtype=np.float32) if ctx.scores else None
    chain = None
    windows = site_windows(site_bin) if windowed else ((0, len(site_bin)),)
    for left, right in windows:
        if windowed:
            ordered = _site_posteriors(posts, site_bin, alleles, gl, config.site_deviation, flips, left, right)
        genotype = _site_posteriors(gposts, site_bin, alleles, gl, config.site_deviation, flips, left, right)
        pg = np.stack([genotype[0], genotype[1] + genotype[2], genotype[3]])
        g = pg.argmax(axis=0)
        if audit is not None:
            audit.argmax(pg, g)
        maximum_genotype = pg.max(axis=0)
        called = maximum_genotype >= tau_g
        if audit is not None:
            audit.difference("genotype", maximum_genotype, tau_g, called)
        assert ordered is not None
        p01, p10 = ordered[1], ordered[2]
        het = called & (g == 1)
        phase_confident = np.maximum(p01, p10) >= tau_p
        phased = het & phase_confident
        if audit is not None:
            audit.difference("phase", np.maximum(p01, p10), tau_p, phase_confident)
        local_calls = calls[left:right]
        hom = called & (g != 1)
        local_calls[hom, 0] = local_calls[hom, 1] = (g[hom] // 2).astype(np.int8)
        one_first = p10 > p01
        if audit is not None:
            audit.difference("orientation", p10, p01, one_first)
        local_calls[het, 0] = np.where(one_first[het], 1, 0)
        local_calls[het, 1] = np.where(one_first[het], 0, 1)
        undetermined[left:right] = het & ~phased
        if scores is not None:
            scores[:, left:right] = np.stack([pg.max(axis=0), np.maximum(p01, p10)]).astype(np.float32)
    if ctx.chain:
        chain = np.full((2, 2, B), np.nan, dtype=np.float64)
        for k in (0, 1):
            if kinds[k] == 1:
                source = post.sum(axis=2 - k)  # [B, n_k]: homologue k's state marginal
                chain[k, 0] = source[:, 0]
                chain[k, 1] = source[:, 1]
    return SampleRelease(calls, undetermined, tuple(set_breaks), scores, chain)


__all__ = ["ReleaseInputs", "release_tables", "label_pair_posterior", "sample_flips", "release_sample"]


# Public producer hook; the leading-underscore spelling remains for existing callers.
bin_orientation = _bin_orientation
