"""Conditional founder-allele refit kernel.

Given the painting, the allele of founder row k at site j maximises the robust
genotype log likelihood of the samples painted as carrying k at j, with each
sample's other painted homologue held at its current allele. Each site is
updated by coordinate ascent over single-row flips and two-row swaps (a fixed
number of sweeps). The minimum gain is a prior per allele: a move is accepted
only when its gain reaches the minimum gain for every cell it changes (one for
a flip, two for a swap). A swap exchanges the different alleles of two rows
painted together at the site: it keeps the genotype of every sample painted
with that pair, so a single flip of either row loses likelihood on those
samples and only the swap corrects an allele carried by the wrong one of the
two rows. Unknown alleles stay unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numba import njit, prange

from ..model.components import ComponentPanel
from ..model.evidence import ContigEvidence
from ..painting.release import ChromosomePainting
from .config import FounderRefitConfig

# Sites per parallel task; each task owns one founder-pair table.
SITE_TILE = 1024


@njit(cache=True)
def _swap_gain(table, alleles, a, b):
    """Log-likelihood change from exchanging the different alleles of called rows a < b.

    The samples painted with the pair {a, b} keep their genotype (their table
    entry cancels), so the change is the sum of each row's change on its other
    carriers: row a takes allele x_b and row b takes x_a.
    """
    xa, xb = alleles[a], alleles[b]
    change = (table[a, a, 2 * xb] - table[a, a, 2 * xa]) + (table[b, b, 2 * xa] - table[b, b, 2 * xb])
    for partner in range(len(alleles)):
        other = alleles[partner]
        if partner == a or partner == b or other < 0:
            continue
        lo, hi = min(a, partner), max(a, partner)
        change += table[lo, hi, xb + other] - table[lo, hi, xa + other]
        lo, hi = min(b, partner), max(b, partner)
        change += table[lo, hi, xa + other] - table[lo, hi, xb + other]
    return change


@njit(cache=True, parallel=True)
def _refit_kernel(calls, first, second, likelihoods, observed, evidence_sites,
                  robustness_epsilon, minimum_gain, sweeps):
    """Coordinate ascent per site on a founder-pair log-likelihood table.

    ``table[a, b, g]`` (a <= b) sums log((1 - eps) P(reads | dosage g) + eps / 3)
    over observed samples painted with the unordered pair {a, b}; a row's
    objective then needs only its partners' current alleles. Pairs whose
    partner allele is unknown contribute nothing, as do unpainted tracks. Each
    sweep tries every single-row flip (accepted at a gain >= minimum_gain),
    then every swap of two rows with different alleles that are painted
    together at the site (``together[a, b]``: some sample painted {a, b} is
    observed there), accepted at a gain >= 2 x minimum_gain because it changes
    two cells. O(L (N + K^2 + P K)) for P co-painted pairs per site
    (P <= min(N, K (K - 1) / 2)), with 1024-site tiles in parallel.
    """
    founders, sites = calls.shape
    samples = first.shape[0]
    refit = calls.copy()
    gain = np.zeros((founders, sites), dtype=np.float64)
    swap_minimum_gain = 2.0 * minimum_gain
    tiles = (sites + SITE_TILE - 1) // SITE_TILE
    for tile in prange(tiles):
        table = np.zeros((founders, founders, 3), dtype=np.float64)
        together = np.zeros((founders, founders), dtype=np.bool_)
        for site in range(tile * SITE_TILE, min(sites, (tile + 1) * SITE_TILE)):
            table[:] = 0.0
            together[:] = False
            column = evidence_sites[site]
            carried = False
            for sample in range(samples):
                a, b = first[sample, site], second[sample, site]
                if a < 0 or b < 0 or not observed[sample, column]:
                    continue
                if a > b:
                    a, b = b, a
                for dosage in range(3):
                    table[a, b, dosage] += math.log(
                        (1.0 - robustness_epsilon) * likelihoods[sample, column, dosage]
                        + robustness_epsilon / 3.0)
                together[a, b] = True
                carried = True
            if not carried:
                continue
            for _ in range(sweeps):
                for row in range(founders):
                    current = refit[row, site]
                    if current < 0:
                        continue
                    ll0 = table[row, row, 0]
                    ll1 = table[row, row, 2]
                    for partner in range(founders):
                        other = refit[partner, site]
                        if partner == row or other < 0:
                            continue
                        lo, hi = min(row, partner), max(row, partner)
                        ll0 += table[lo, hi, other]
                        ll1 += table[lo, hi, other + 1]
                    delta = ll1 - ll0 if current == 0 else ll0 - ll1
                    if delta > 0.0 and delta >= minimum_gain:
                        refit[row, site] = 1 - current
                        gain[row, site] = delta
                # Swaps only (class-B fix B1; 92b3daf flipped any two called rows
                # together on a summed gain >= minimum_gain, so rows never painted
                # together, or with equal alleles, changed on less than the prior
                # of each cell and identical rows could end with different calls).
                for a in range(founders):
                    for b in range(a + 1, founders):
                        if not together[a, b]:
                            continue
                        xa, xb = refit[a, site], refit[b, site]
                        if xa < 0 or xb < 0 or xa == xb:
                            continue
                        delta = _swap_gain(table, refit[:, site], a, b)
                        if delta > 0.0 and delta >= swap_minimum_gain:
                            refit[a, site] = xb
                            refit[b, site] = xa
                            gain[a, site] = delta
                            gain[b, site] = delta
    return refit, gain


def refit_alleles(calls: np.ndarray, first: np.ndarray, second: np.ndarray,
                  likelihoods: np.ndarray, observed: np.ndarray, evidence_sites: np.ndarray, *,
                  robustness_epsilon: float, minimum_gain_nats: float,
                  coordinate_sweeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Refit a K-by-L allele matrix (-1 unknown, never filled) given painted rows.

    ``first``/``second`` int16[N, L]: founder rows of the two painted tracks at
    each panel site (-1 unknown); ``likelihoods`` float64[N, L_e, 3]
    normalised and ``observed`` bool[N, L_e] on the evidence site axis;
    ``evidence_sites`` int64[L] maps panel sites to it. Returns the refit
    matrix and the accepted gain per cell (zero where unchanged; a cell flipped
    twice keeps its last gain).
    """
    return _refit_kernel(
        np.ascontiguousarray(calls, dtype=np.int8), np.ascontiguousarray(first, dtype=np.int16),
        np.ascontiguousarray(second, dtype=np.int16), likelihoods,
        np.ascontiguousarray(observed, dtype=np.bool_),
        np.ascontiguousarray(evidence_sites, dtype=np.int64),
        float(robustness_epsilon), float(minimum_gain_nats), int(coordinate_sweeps))


@dataclass(frozen=True)
class ComponentRefit:
    """One painted component after the refit.

    panel: the painted panel with ``released`` replaced by the refit alleles and
        ``inference`` = the refit alleles on the frozen snapshot's called cells
        (-1 elsewhere: the snapshot's unknown cells stay unknown).
    changed_rows int32[M], changed_sites int64[M] (panel site index),
    gain_nats float64[M]: released cells that changed, with their accepted gain.
    """

    panel: ComponentPanel
    changed_rows: np.ndarray
    changed_sites: np.ndarray
    gain_nats: np.ndarray


@dataclass(frozen=True)
class FounderRefit:
    """Refit product of one chromosome (record ``founder_refit``).

    excluded_samples: sample ids whose reads the sample QC keeps out of this
    contig's founder evidence (and so out of the refit).
    """

    contig: str
    sample_ids: tuple[str, ...]
    components: tuple[ComponentRefit, ...]
    excluded_samples: tuple[str, ...]

    @property
    def panels(self) -> tuple[ComponentPanel, ...]:
        """The refit ComponentPanels in chromosome order."""
        return tuple(component.panel for component in self.components)


def refit_chromosome(painting: ChromosomePainting, evidence: ContigEvidence,
                     config: FounderRefitConfig) -> FounderRefit:
    """Refit every painted component of one chromosome; the painting is unchanged.

    ``painting`` must carry its panels (painting.records.load(..., panels=True));
    ``evidence`` is the contig's evidence (its founder view is used here).
    """
    founder = evidence.founder_view()
    components = []
    for component in painting.components:
        panel = component.panel
        sites = panel.site_indices()
        labels = component.site_labels(panel.positions)
        refit_calls, gain = refit_alleles(
            panel.released, labels[:, 0], labels[:, 1], founder.likelihoods, founder.observed,
            sites, robustness_epsilon=painting.config.robustness_epsilon,
            minimum_gain_nats=config.minimum_gain_nats,
            coordinate_sweeps=config.coordinate_sweeps)
        del labels
        rows, columns = np.nonzero(refit_calls != panel.released)
        components.append(ComponentRefit(
            panel=replace(panel, released=refit_calls,
                          inference=np.where(panel.inference >= 0, refit_calls, -1).astype(np.int8)),
            changed_rows=rows.astype(np.int32), changed_sites=columns.astype(np.int64),
            gain_nats=gain[rows, columns]))
    excluded = tuple(sample for sample, flag in zip(evidence.sample_ids, evidence.founder_excluded)
                     if flag)
    return FounderRefit(contig=painting.contig, sample_ids=tuple(painting.sample_ids),
                        components=tuple(components), excluded_samples=excluded)


def summary(refit: FounderRefit) -> dict:
    """Small per-chromosome summary (log line, stage summary table)."""
    gains = (np.concatenate([component.gain_nats for component in refit.components])
             if refit.components else np.zeros(0))
    return {
        "contig": refit.contig, "components": len(refit.components),
        "founder_rows": sum(component.panel.founders for component in refit.components),
        "sites": sum(len(component.panel.positions) for component in refit.components),
        "called_alleles": sum(int(np.sum(component.panel.released >= 0))
                              for component in refit.components),
        "changed_alleles": int(len(gains)),
        "changed_inference_alleles": sum(
            int(np.sum(component.panel.inference[component.changed_rows,
                                                 component.changed_sites] >= 0))
            for component in refit.components),
        "changed_sites": sum(len(np.unique(component.changed_sites))
                             for component in refit.components),
        "median_changed_gain_nats": float(np.median(gains)) if len(gains) else None,
        "excluded_samples": list(refit.excluded_samples),
    }
