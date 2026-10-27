"""Release of a selected panel's alleles and its unknown-copy summary.

A founder allele is released when its one-bit conditional probability (the exact marginal
likelihood ratio of flipping it, every other bit and pi fixed, with equal prior odds) reaches
``min_call_probability`` and the samples whose own evidence favours it (their per-sample exact
gain of flipping it is below -``favourable_margin``) include ``min_supporters`` carriers: as many
samples each more likely than not carrying the founder at the site, or that much expected carrier
mass (the mass alone sits just below 2 for a founder with exactly two near-certain carriers; the
count alone would withhold diffusely supported sites). Released alleles are the panel's own.

The unknown-copy summary for assembly: per sample the largest marginal MAP number of unknown
copies over its observed sites (2 without depth) and their fraction over samples with depth.
Release 2.1: a site's MAP count must win by more than ``unknown_count_band`` relative posterior
mass, else the smaller count (design A-7: the count feeds completion's all-or-nothing veto, and
its masses can tie exactly); the band decisions are counted ("ls.unknown.band") and, with the
valley census on, the samples' smallest relative gaps near zero are recorded.

Release 2.12, context witness: the probability test is unchanged, but the replication test (``min_supporters``) can
be met by the context refit. The block posterior splits the carriers of near-identical rows by frequency, so a row
the context claims kept fewer than ``min_supporters`` carriers of its own and was withheld wholesale although its
alleles are witnessed (context snaps made such rows common). An allele that passes the probability test but not the
support test is released when the row is claimed by an assembled context founder in the panel's shortest description
(objective.ContextFounders.matched), the allele equals that founder's call, and the context refit itself released the
call for a founder carrying this local row (ContextProposal.witnessed: site posterior ``posterior_threshold`` with
``min_support`` carriers identified by linkage from the flanking blocks; not a retained input call or a completion
fill). The refit reads the same block's reads with that carrier assignment: conditional reuse of the evidence, not
independent confirmation.

Release 2.1: ``favourable_margin`` is the ``cell`` tie quantum, a band around the exact zero gain
of symmetric-evidence cells (1 REF + 1 ALT read: GL(0/0) == GL(1/1)), which are not favourable
inside it (design A-4, A-7); the band decisions are counted ("ls.favourable.band") and, with the
valley census on, the carrier cells' gains near zero are recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ..maths.ties import census, census_enabled
from ..model.local_panels import LocalPanel, unknown_copy_mass
from .path_model import release_statistics

# Census window of the favourable band (carrier cells with |exact gain| up to this are recorded).
FAVOURABLE_CENSUS_WINDOW = 1e-3
# Census window of the unknown-count band (samples whose smallest relative gap between a site's two
# largest count masses is at most this are recorded).
UNKNOWN_CENSUS_WINDOW = 1e-3


@dataclass(frozen=True)
class PathRelease:
    """calls int8[K, L_kept] (-1 withheld); probability float64[K, L_kept] of the current
    allele; supporters int64[K, L_kept] favourable carriers by count; unknown_copy_fraction:
    posterior unknown-state occupancy over all copies (diagnostic); has_depth bool[N];
    unknown_slots int8[N]; unknown_mass float; favourable_band_cells: observed carrier cells whose
    gain lies inside the favourable band (diagnostic, release 2.1; 0 in older records)."""

    calls: np.ndarray
    probability: np.ndarray
    supporters: np.ndarray
    unknown_copy_fraction: float
    has_depth: np.ndarray
    unknown_slots: np.ndarray
    unknown_mass: float
    favourable_band_cells: int = 0
    context_cells: int = 0


def context_witnessed(panel, context, witness):
    """bool[K, L]: cells whose allele equals the call of the context founder claiming the row in the panel's
    shortest description, where the context refit released that call (``witness``: one row per assembled founder,
    aligned with the founders ``context`` was built from)."""
    panel = np.asarray(panel, dtype=np.int64)
    out = np.zeros(panel.shape, dtype=np.bool_)
    for founder, row in context.matched(panel):
        out[row] |= (panel[row] == context.founders[founder]) & witness[context.source[founder]]
    return out


def release(fit, prepared, config, ties=None, census_key=None, context=None, witness=None) -> PathRelease:
    """The PathRelease of a selected fit. ``ties`` (maths.ties.TieCounter) counts the favourable band
    and the unknown-count band (one decision per release each; inside: some observed carrier cell's
    gain in the favourable band, some observed site's count lowered by the unknown-count band);
    ``census_key`` records the carrier cells' gains within FAVOURABLE_CENSUS_WINDOW of zero and the
    samples' smallest count gaps within UNKNOWN_CENSUS_WINDOW when the census is on. ``context``
    (objective.ContextFounders) and ``witness`` (the context refit's released mask per assembled founder) enable
    the context-witness rescue of the module docstring."""
    collect = census_key is not None and census_enabled()
    statistics = release_statistics(fit.panel, fit.frequencies, prepared, config.favourable_margin,
                                    config.unknown_count_band, keep_gains=collect)
    if ties is not None:
        # One decision per release; inside the tie when some observed carrier cell's gain lies in the
        # band (their number is PathRelease.favourable_band_cells).
        ties.note("ls.favourable.band", inside_tie=statistics["band_cells"] > 0)
        ties.note("ls.unknown.band", inside_tie=statistics["unknown_band_sites"] > 0)
    if collect:
        gains = statistics["sample_gains"]
        for sample, founder, site in np.argwhere(np.abs(gains) <= FAVOURABLE_CENSUS_WINDOW):
            census("ls.favourable.band", f"{census_key}/n{sample}/a{founder}/s{site}",
                   (0.0, float(gains[sample, founder, site])))
        for sample in np.flatnonzero(statistics["unknown_gaps"] <= UNKNOWN_CENSUS_WINDOW):
            census("ls.unknown.band", f"{census_key}/n{sample}",
                   (0.0, float(statistics["unknown_gaps"][sample])))
    probability = expit(-statistics["conditional_flip_gains"])
    supporters, mass = statistics["supporters"], statistics["carrier_mass"]
    confident = ((probability >= config.min_call_probability)
                 & (np.maximum(supporters, mass) >= config.min_supporters))
    context_cells = 0
    need = (probability >= config.min_call_probability) & ~confident
    if need.any() and context is not None and witness is not None and len(context):
        rescue = context_witnessed(fit.panel, context, witness) & need
        context_cells = int(rescue.sum())
        confident |= rescue
    occupancy = statistics["occupancy"]
    has_depth = statistics["has_depth"]
    slots = statistics["unknown_slots"]
    return PathRelease(
        calls=np.where(confident, fit.panel, -1).astype(np.int8), probability=probability,
        supporters=supporters, unknown_copy_fraction=float(occupancy[-1].sum()) / float(occupancy.sum()),
        has_depth=has_depth, unknown_slots=slots, unknown_mass=unknown_copy_mass(slots, has_depth),
        favourable_band_cells=int(statistics["band_cells"]), context_cells=context_cells)


def local_panel(block: int, site_start: int, positions, kept, fit, path_release: PathRelease
                ) -> LocalPanel:
    """The LocalPanel of a selected fit: released calls, the fit's allele probabilities and the
    favourable carrier counts at the discovery sites; -1 / 0.5 / 0 elsewhere."""
    k = len(fit.panel)
    length = len(positions)
    calls = np.full((k, length), -1, dtype=np.int8)
    calls[:, kept] = path_release.calls
    alt_probability = np.full((k, length), .5)
    alt_probability[:, kept] = fit.allele_probability
    support = np.zeros((k, length))
    support[:, kept] = path_release.supporters
    return LocalPanel(block=int(block), site_start=int(site_start),
                      positions=np.ascontiguousarray(positions, dtype=np.int64),
                      kept=np.ascontiguousarray(kept, dtype=np.bool_), calls=calls,
                      alt_probability=alt_probability, support=support,
                      has_depth=path_release.has_depth, unknown_slots=path_release.unknown_slots,
                      unknown_mass=path_release.unknown_mass)
