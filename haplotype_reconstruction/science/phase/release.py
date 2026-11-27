"""Release of one component: calls, phase, phase sets and their uncertainty from exact block posteriors.

For each sample, every other sample fixed at the converged MAP, the block's exact forward-backward
(kernels.forward_backward) gives posteriors of its ordered label pair per bin. The site model turns them into per-site
ordered allele posteriors (kernels.site_posteriors): given its label pair, each homologue carries its label's panel
allele except for a flip with that allele's conditional uncertainty (allele_evidence.py: the sample's direct
own-read term is removed with the fitted carriers, panel and empirical prior held fixed; indirect own-read
influence on those fitted quantities remains),
the pair departs from the alleles with probability eps (the run's outlier rate, estimated at the painting calibration:
a redraw from the site's cohort genotype frequencies, science.shared.observation), and its own reads at the site weigh
the alternatives.

- Genotypes come from the unanchored posterior (the homologue-order gauge does not change unordered genotypes). A
  genotype is called where its posterior reaches ``genotype_posterior``; otherwise both alleles are unknown.
- Phase of a sample with a linked parent is anchored by parental origin (homologue k is the slot assigned to the
  slot-k parent: its copy where the chain's source is 0/1).
  A sample without a linked parent but with children is anchored by its children's chains. A sample with neither is a
  gauge sample: its chromosome is partitioned at hard cuts (forced breaks, below) and at every heterozygous bin where,
  given the current partition's anchor, the orientation of its alleles is uncertain: the median over the bin's
  witness sites of max(P01, P10) / (P01 + P10) falls below ``set_posterior`` (stricter than ``phase_posterior``: a
  wrong continuation misphases the rest of the set).
  This set statistic uses its own site model, ``set_site_deviation`` against a uniform genotype, not the run's outlier
  rate, which would cap every common heterozygote below ``set_posterior`` (parameters.py). A witness is a site
  confidently heterozygous (P(heterozygous) >= ``genotype_posterior``) under that model that the labels alone (the
  same site model with no departure) do not make confidently homozygous; its certainty is the set model's
  (``set_orientation``). A site heterozygous only through the departure has P01 = P10 under every frame and is no
  evidence about the frame.
  Uncertainty about which labels a sample carries does not break its phase where the
  candidate labels carry the same alleles at its sites (the orientation is then still determined). Each partition is
  anchored at its first confidently heterozygous bin (P(heterozygous) >= ``phase_posterior``) by the gauge "homologue
  0 carries the label ranked first", labels ranked by how closely their panel alleles at the bin match homologue 0 of
  the bin's most probable pair (``anchor_mask``; no decoded label is assumed; where that pair's alleles never differ
  in the bin, by label index); a partition without such a bin releases no phase.
- Forced breaks (``break_bins``, chosen by the stage): a sample without a linked parent breaks its phase sets at its
  painting tie boundaries (the painting's track order there is a convention), and every sample breaks at the junction
  candidates where it carries both junction labels, unless the parental homologue it copies there is pinned by the
  parent's own reads (junctions.withheld_joins). A break is recorded at its own bin, never moved to the next anchor,
  and no phase is released across it. A sample anchored by its children takes its orientation from their chains on
  both sides of a painting tie, so it keeps a tie break (one that is not also a withheld join) only where that
  orientation is uncertain beside it: where the orientation certainty above, under its children-anchored posterior,
  is below ``set_posterior`` at the nearest bin with a witness site on either side (``_bridged_ties``). Each solve
  decides for itself, and the agreement keeps a break that either solve keeps.
- A child with two fitted incoming parents breaks at the right anchor of a tied transmission passage when the
  relative-order odds fall below the usual set-continuation odds (FP20). The test only adds break bins.
- A heterozygous call is released phased where its ordered posterior reaches ``phase_posterior``; otherwise it is
  released unphased (frame_undetermined bits 0 and 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.phase import ReleaseTables, SampleRelease
from ...contracts.roles import record_field
from . import release_kernels as kernels
from .allele_evidence import own_terms
from .model import LabelModel
from .parameters import RELEASE_ADVANCE_TERMS, RELEASE_WINDOW, PhaseParameters
from .transmission_passages import transmission_breaks
from .window import site_windows


@dataclass(frozen=True)
class ReleaseInputs:
    """One sample's site release inputs (FP14-FP19), borrowed without mutation.

    Site axes contain only released component sites. GLs are float32[L,3] to match the evidence input precision;
    all uncertainty and decision arithmetic is float64.
    """

    config: PhaseParameters = record_field("state", units="1", dtype="PhaseParameters", stored=False)
    site_bin: np.ndarray = record_field("state", units="1", dtype="int64", stored=False)
    site_alleles: np.ndarray = record_field("state", units="1", dtype="int8", stored=False)
    site_gl: np.ndarray = record_field("evidence", units="1", dtype="float32", stored=False)
    site_cohort: np.ndarray = record_field("state", units="probability", dtype="float64", stored=False)
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
    # The solve's per-window contiguous copies of site_alleles and flips (shared_windows), shared by all samples.
    shared_windows: dict | None = record_field("diagnostic", units="1", dtype="dict | None", stored=False, default=None)


def shared_windows(site_bin: np.ndarray, alleles: np.ndarray, flips: np.ndarray) -> dict:
    """{(lo, hi): (alleles [S, w], flips [S, w])} contiguous, for the release's site windows (RELEASE_WINDOW)."""
    return {
        (lo, hi): (np.ascontiguousarray(alleles[:, lo:hi]), np.ascontiguousarray(flips[:, lo:hi]))
        for lo, hi in site_windows(site_bin, window=RELEASE_WINDOW)
    }


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
    return kernels.label_pairs(post, labs0, labs1, int(S))


def bin_orientation(ordered, site_bin, tau_g, B):
    """Per-bin median orientation certainty over confidently heterozygous sites; NaN without such a site."""
    return kernels.orientation(ordered, site_bin, float(tau_g), int(B))


def set_orientation(ordered, label_only, site_bin, tau_g, B):
    """The set statistic per bin: bin_orientation over the bin's witness sites, float64 [B] (NaN without one).

    ordered float64 [4, m]: the sites' ordered allele posteriors under the set statistic's site model
    (``set_site_deviation`` against a uniform genotype); label_only [4, m]: the same posteriors with no departure
    (epsilon 0), under which a site is heterozygous only where a candidate label pair, or a flip of a panel allele,
    makes it so. A confidently heterozygous site (P(heterozygous) >= tau_g under the set model) is a witness unless
    the labels alone make it confidently homozygous (P(00) + P(11) >= tau_g at epsilon 0): then it is heterozygous
    only through the departure, has P01 = P10 under every frame, carries no evidence about the frame, and in a bin
    with one or two confident sites would decide the median. Every other confident site stays, with its set-model
    certainty, so a label pair whose alleles cannot produce the reads still enters at 50/50 wherever some candidate
    pair explains part of the heterozygote. Where no confident site is label-homozygous this is
    bin_orientation(ordered). ``ordered`` is overwritten (both callers own it).
    """
    homozygous = label_only[0] + label_only[3] >= tau_g
    ordered[1, homozygous] = 0.0
    ordered[2, homozygous] = 0.0
    return bin_orientation(ordered, site_bin, tau_g, B)


def anchor_mask(pair_posterior, labels0, labels1, alleles):
    """The anchor mask bool [n0, n1] of a gauge partition: homologue 0 carries the label ranked first.

    pair_posterior float64 [S, S]: the anchor bin's unanchored ordered label-pair posterior; labels0 [n0], labels1
    [n1]: the block states' labels there; alleles int8 [S, k]: the panel alleles at the bin's k sites (-1 unknown).
    Labels are ranked by how closely their alleles match homologue 0 of the bin's most probable pair (x, y) taken in
    label order (x < y): over the bin's sites where a_x and a_y are known and differ, the number of sites at which a
    label carries a_x minus the number at which it carries a_y (an unknown allele counts for neither). Equal scores
    keep x first among the labels that score as x and y last among those that score as y (a label carrying a_x at
    every such site is not put before x, which would split x across homologues), otherwise label index. The ranking
    is a strict total order, so the mask keeps one ordering of every unordered pair of distinct labels: a gauge, which
    leaves unordered posteriors and genotype calls unchanged, and (x, y) keeps its label order. Ranking by the panel
    alleles fixes the allele frame where the bin's posterior splits between pairs that share a label in swapped
    sorted positions, which label order would put on opposite homologues. Where a_x and a_y never differ in the bin
    the alleles give no frame and the mask is label order, "homologue 0 carries the smaller label"; where no
    posterior mass sits on pairs that the ranking orders against label order, it selects the same mass as that mask.
    """
    S = pair_posterior.shape[0]
    x, y = divmod(int(np.argmax(np.triu(pair_posterior, 1))), S)
    first, second = alleles[x], alleles[y]
    informative = (first >= 0) & (second >= 0) & (first != second)
    if not informative.any():
        return labels0[:, None] < labels1[None, :]
    panel = alleles[:, informative]
    resemblance = (panel == first[informative]).sum(axis=1) - (panel == second[informative]).sum(axis=1)
    tie = np.arange(S)
    tie[x], tie[y] = -1, S  # x scores the most and y the least possible: first and last within their scores
    rank = np.empty(S, dtype=np.int64)
    rank[np.lexsort((tie, -resemblance))] = np.arange(S)
    return rank[labels0][:, None] < rank[labels1][None, :]


def _site_posteriors(post, site_bin, alleles, gl, cohort, epsilon, flips, lo=0, hi=None, shared=None):
    """kernels.site_posteriors over the sites [lo, hi) with the sample's flip arrays ``flips`` (flips [S, m],
    own_state [2, m], own_flip [2, m]); ``shared``: the solve's window copies of alleles and flips[0]."""
    hi = len(site_bin) if hi is None else hi
    if lo == 0 and hi == len(site_bin):
        return kernels.site_posteriors(post, site_bin, alleles, gl, cohort, epsilon, *flips)
    window = None if shared is None else shared.get((lo, hi))
    if window is None:
        window = (np.ascontiguousarray(alleles[:, lo:hi]), np.ascontiguousarray(flips[0][:, lo:hi]))
    return kernels.site_posteriors(
        post,
        site_bin[lo:hi],
        window[0],
        gl[lo:hi],
        np.ascontiguousarray(cohort[lo:hi]),
        epsilon,
        window[1],
        *(np.ascontiguousarray(a[:, lo:hi]) for a in flips[1:]),
    )


def sample_flips(model, n, ctx):
    """(flips [S, m], own_state [2, m], own_flip [2, m]) of sample n (kernels.site_posteriors): the solve's flip
    probabilities, with the labels of n's own MAP pair recomputed without n's reads (allele_evidence.own_terms)."""
    site_bin, alleles = ctx.site_bin, ctx.site_alleles
    labels = np.ascontiguousarray(model.X[n][:, site_bin])
    state, flip = own_terms(
        labels, alleles, ctx.site_gl, ctx.robust_epsilon, bool(ctx.excluded), ctx.allele_evidence, ctx.logit_prior,
        ctx.site_cohort,
    )
    return ctx.flips, state, flip


def _piece_orientation(lp, site_bin, alleles, gl, cohort, epsilon, flips, tau_g, lo_bin, hi_bin, windowed=True):
    """set_orientation of the bins [lo_bin, hi_bin), float64 [hi_bin - lo_bin], under their label-pair posteriors
    ``lp`` [hi_bin - lo_bin, S, S] (site model: departure ``epsilon`` against ``cohort``), using whole-bin site
    windows as the workspace bound."""
    bins = hi_bin - lo_bin
    lo = int(np.searchsorted(site_bin, lo_bin, side="left"))
    hi = int(np.searchsorted(site_bin, hi_bin, side="left"))
    orientation = np.full(bins, np.nan)
    if hi <= lo:
        return orientation
    for left, right in site_windows(site_bin, lo, hi, window=RELEASE_WINDOW) if windowed else ((lo, hi),):
        local = site_bin[left:right] - lo_bin
        window_alleles = np.ascontiguousarray(alleles[:, left:right])
        window_cohort = np.ascontiguousarray(cohort[left:right])
        window_flips = tuple(np.ascontiguousarray(a[:, left:right]) for a in flips)
        window_gl = gl[left:right]
        ordered = kernels.site_posteriors(lp, local, window_alleles, window_gl, window_cohort, epsilon, *window_flips)
        label_only = kernels.site_posteriors(lp, local, window_alleles, window_gl, window_cohort, 0.0, *window_flips)
        values = set_orientation(ordered, label_only, local, tau_g, bins)
        orientation[local[0] : local[-1] + 1] = values[local[0] : local[-1] + 1]
    return orientation


def _bridged_ties(ties, orientation, tau_s):
    """The painting-tie cuts (bins) that a sample anchored by its children bridges: those where the orientation
    certainty (``orientation``, set_orientation under its children-anchored posterior) at the nearest bin with a
    witness site on each side (before the cut's bin, and at or after it) reaches tau_s. A cut without such a bin on one
    side is kept."""
    informative = np.flatnonzero(~np.isnan(orientation))
    out = set()
    for c in ties:
        k = int(np.searchsorted(informative, c, side="left"))
        if 0 < k < len(informative):
            value = min(orientation[informative[k - 1]], orientation[informative[k]])
            bridged = value >= tau_s
            if bridged:
                out.add(int(c))
    return out


def release_sample(model: LabelModel, n: int, ctx: ReleaseInputs, *, map_work=None) -> SampleRelease:
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
    The optional ordered map runs independent hard-cut intervals and final site
    windows. Each item writes only its own bins and uses full-chain posteriors.
    """
    config = ctx.config
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
    cohort = ctx.site_cohort
    uniform = np.full_like(cohort, 1.0 / 3.0)  # the set statistic's genotype law (config.set_site_deviation)
    flips = sample_flips(model, n, ctx)
    map0, map1 = model.X[n, 0], model.X[n, 1]
    het_bins = np.flatnonzero(map0 != map1)
    partitioned = gauge and len(het_bins) > 0
    if partitioned:
        # The anchored posteriors below reuse the unanchored chain's messages.
        alpha, beta = kernels.chain_messages(em, T0, T1, TG)
        post = kernels.posterior_bins(alpha, beta)
    else:
        post = kernels.forward_backward(em, T0, T1, TG, -1, np.ones((1, 1), dtype=np.bool_))
    gposts = label_pair_posterior(post, labs0, labs1, S)
    cuts = sorted({int(b) for b in forced if 0 < b < B})
    set_breaks = []
    if partitioned:
        posts = np.zeros((B, S, S))  # bins of a partition without an anchor: no phase
        het_conf = 1.0 - np.einsum("bxx->b", gposts)  # P(the label pair is heterozygous)
        candidates = het_bins[het_conf[het_bins] >= tau_p]  # ascending
        n0, n1, G = em.shape[1:]
        advance = max(1, RELEASE_ADVANCE_TERMS // (n0 * n1 * (n0 + n1) * G))

        def anchored(first, messages, lo, hi):
            """Label-pair posteriors [hi - lo, S, S] of the bins [lo, hi) of the anchored chain from its forward
            (``first``) and backward (``messages``) messages there."""
            return label_pair_posterior(kernels.posterior_bins(first, messages), labs0[lo:hi], labs1[lo:hi], S)

        def partition(bounds):
            # Hard cuts fix independent release intervals. Each anchored posterior is that of the complete chain
            # restricted at the anchor bin, with the same anchor search and continuation tests inside its interval;
            # it is evaluated where the interval reads it: back to the interval's start, and forward piece by piece
            # until the continuation test fails (the restriction changes neither alpha before the anchor nor beta
            # at and after it, and a bin's orientation certainty depends on that bin's posterior alone).
            start, hard = bounds
            breaks = []
            while start < hard:
                if start > 0:
                    breaks.append(start)
                following = int(np.searchsorted(candidates, start))
                if following == len(candidates) or candidates[following] >= hard:
                    break
                anchor = int(candidates[following])
                first_site, end_site = np.searchsorted(site_bin, (anchor, anchor + 1))
                mask = anchor_mask(gposts[anchor], labs0[anchor], labs1[anchor], alleles[:, first_site:end_site])
                if anchor > start:
                    before = kernels.backward_to(beta[anchor], em, T0, T1, TG, start, anchor, mask)
                    posts[start:anchor] = anchored(alpha[start:anchor], before, start, anchor)
                last = kernels.restricted(alpha[anchor], mask)
                posts[anchor] = anchored(last[None], beta[anchor : anchor + 1], anchor, anchor + 1)[0]
                stop, lo = hard, anchor + 1
                # When sets fragment, most of a large initial piece would be
                # recomputed at every new anchor. Grow after successful tests.
                width = 1
                while lo < hard:
                    hi = min(hard, lo + width)
                    forward = kernels.forward_from(last, em, T0, T1, TG, lo, hi)
                    lp = anchored(forward, beta[lo:hi], lo, hi)
                    orientation = _piece_orientation(
                        lp, site_bin, alleles, gl, uniform, config.set_site_deviation, flips, tau_g, lo, hi, windowed
                    )
                    tested = het_bins[np.searchsorted(het_bins, lo) : np.searchsorted(het_bins, hi)]
                    failing = next(
                        (int(b) for b in tested if het_conf[b] >= tau_g and orientation[b - lo] < tau_s), None
                    )
                    if failing is not None:
                        posts[lo:failing] = lp[: failing - lo]
                        stop = failing
                        break
                    posts[lo:hi] = lp
                    lo, last = hi, forward[-1]
                    width = min(advance, 2 * width)
                start = stop
            return breaks

        boundaries = [0, *cuts, B]
        intervals = list(zip(boundaries[:-1], boundaries[1:], strict=True))
        released = (
            map_work(partition, intervals)
            if map_work is not None and len(intervals) > 1
            else [partition(bounds) for bounds in intervals]
        )
        set_breaks = [edge for interval in released for edge in interval]
        alpha = beta = None  # the chain messages are not read below
    else:
        posts = gposts
        set_breaks = cuts
    ties = ctx.tie_bins
    orientation = np.full(B, np.nan) if children and not anchored_by_parent and len(ties) else None
    calls = np.full((len(site_bin), 2), -1, dtype=np.int8)
    undetermined = np.zeros(len(site_bin), dtype=bool)
    scores = np.empty((2, len(site_bin)), dtype=np.float32) if ctx.scores else None
    shared = ctx.shared_windows

    def release_sites(bounds):
        left, right = bounds
        ordered = _site_posteriors(
            posts, site_bin, alleles, gl, cohort, ctx.robust_epsilon, flips, left, right, shared=shared
        )
        # With no gauge partition the unanchored and ordered release use the same site posteriors.
        genotype = ordered
        if partitioned:
            genotype = _site_posteriors(
                gposts, site_bin, alleles, gl, cohort, ctx.robust_epsilon, flips, left, right, shared=shared
            )
        if orientation is not None and right > left:
            bins = site_bin[left:right]
            first, last = int(bins[0]), int(bins[-1]) + 1
            # Tie bridging uses the gauge set statistic: its site model and its witnesses.
            oriented = _site_posteriors(
                posts, site_bin, alleles, gl, uniform, config.set_site_deviation, flips, left, right, shared=shared
            )
            label_only = _site_posteriors(posts, site_bin, alleles, gl, uniform, 0.0, flips, left, right, shared=shared)
            orientation[first:last] = set_orientation(oriented, label_only, bins - first, tau_g, last - first)
        kernels.site_calls(
            genotype, ordered, tau_g, tau_p, calls[left:right], undetermined[left:right],
            None if scores is None else scores[:, left:right],
        )

    # Windows contain complete bins, so every output slice and median is independent.
    windows = (
        tuple(site_windows(site_bin, window=RELEASE_WINDOW))
        if windowed or map_work is not None
        else ((0, len(site_bin)),)
    )
    if map_work is not None and len(windows) > 1:
        map_work(release_sites, windows)
    else:
        for bounds in windows:
            release_sites(bounds)
    if orientation is not None:
        bridged = _bridged_ties(ties, orientation, tau_s)
        set_breaks = [c for c in cuts if c not in bridged]
    passage_cuts = transmission_breaks(model, n, ctx, flips)
    if passage_cuts:
        cut_bins = np.zeros(B, dtype=bool)
        cut_bins[set_breaks] = True
        cut_bins[list(passage_cuts)] = True
        set_breaks = np.flatnonzero(cut_bins).tolist()
    chain = None
    if ctx.chain:
        chain = np.full((2, 2, B), np.nan, dtype=np.float64)
        for k in (0, 1):
            if kinds[k] == 1:
                source = post.sum(axis=2 - k)  # [B, n_k]: homologue k's state marginal
                chain[k, 0] = source[:, 0]
                chain[k, 1] = source[:, 1]
    return SampleRelease(calls, undetermined, tuple(set_breaks), scores, chain)


__all__ = [
    "ReleaseInputs",
    "release_tables",
    "label_pair_posterior",
    "sample_flips",
    "release_sample",
    "bin_orientation",
    "set_orientation",
    "anchor_mask",
]
