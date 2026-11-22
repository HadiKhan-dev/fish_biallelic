"""Refit the painted founder panels without changing their released painting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ...contracts.likelihoods import gather_likelihoods
from ...contracts.refit import ComponentRefit, FounderRefit
from ...contracts.roles import record_field
from ...core.errors import InvariantError
from ..shared.observation import mix_likelihoods
from .parameters import CARRIER_CLASS_EDGES, RefitParameters
from .refit import AcceptancePriors, class_priors, refit_alleles, start_gain_histogram

if TYPE_CHECKING:
    from ...contracts.evidence import ContigEvidence
    from ...contracts.painting import Painting


def _components(painting: Painting, evidence: ContigEvidence, config: RefitParameters):
    """(units, inputs): the painted components with their panels and sites, and the per-component reader of labels,
    mixed likelihoods and observed cells (the painting's outlier rate)."""
    founder = evidence.founder_view()
    units = []
    for component in painting.components:
        panel = component.panel
        if panel is None:
            raise InvariantError("refit requires the painted component panel")
        units.append((component, panel, panel.site_indices()))

    def inputs(component, panel, sites):
        labels = component.site_labels(panel.positions)
        # The component's mixed likelihoods at the painting's outlier rate (science.shared.observation).
        likelihoods = mix_likelihoods(
            gather_likelihoods(founder.likelihoods, sites), founder.cohort_frequencies[sites], painting.robustness_epsilon
        )
        return labels, likelihoods, np.ascontiguousarray(founder.observed[:, sites])

    return units, inputs


def chromosome_histogram(painting: Painting, evidence: ContigEvidence, config: RefitParameters) -> np.ndarray:
    """Pass 1 (ADVICE2 12.2): the contig's start-state single-flip gain histogram by carrier class, int64 counts (a sum
    over contigs is exact in any order). The run's class priors pool every analysed contig's histogram (N0run, the arm
    frozen by gate G7', ADVICE2 16.2: one coefficient per class and run)."""
    units, inputs = _components(painting, evidence, config)
    histogram = None
    for component, panel, sites in units:
        labels, likelihoods, observed = inputs(component, panel, sites)
        part = start_gain_histogram(
            panel.released, labels[:, 0], labels[:, 1], likelihoods, observed, np.arange(len(sites), dtype=np.int64)
        )
        histogram = part if histogram is None else histogram + part
        del labels, likelihoods, observed
    return histogram if histogram is not None else np.zeros((len(CARRIER_CLASS_EDGES) + 1, 1), dtype=np.int64)


def run_priors(histograms) -> AcceptancePriors:
    """The run's class priors from the analysed contigs' histograms (summed; pooled run-wide by the 4,096-cell rule)."""
    histograms = [np.asarray(h) for h in histograms]
    if not histograms:
        return class_priors(np.zeros((len(CARRIER_CLASS_EDGES) + 1, 1), dtype=np.int64))
    width = max(h.shape[1] for h in histograms)
    total = np.zeros((histograms[0].shape[0], width), dtype=np.int64)
    for h in histograms:
        total[:, : h.shape[1]] += h.astype(np.int64)
    return class_priors(total)


def refit_chromosome(
    painting: Painting, evidence: ContigEvidence, config: RefitParameters, *, reverse: bool = False,
    priors: AcceptancePriors | None = None,
) -> FounderRefit:
    """Refit every painted component of one chromosome; the painting is unchanged.

    ``painting`` must carry its panels (painting.records.load(..., assembly=assembly));
    ``evidence`` is the contig's evidence (its founder view is used here). ``priors``: the run's class priors
    (run_priors over every analysed contig); without them the contig's own histogram (a one-contig run).
    """
    units, inputs = _components(painting, evidence, config)
    if priors is None:
        priors = run_priors([chromosome_histogram(painting, evidence, config)])
    classes = len(priors.cells)
    changed_by_class = np.zeros(classes, dtype=np.int64)
    filled_by_class = np.zeros(classes, dtype=np.int64)
    guard_sites = 0
    certificate = np.zeros(3, dtype=np.int64)
    components = []
    for component, panel, sites in units:
        labels, likelihoods, observed = inputs(component, panel, sites)
        refit_calls, gain, guard, cell_class, residual = refit_alleles(
            panel.released,
            labels[:, 0],
            labels[:, 1],
            likelihoods,
            observed,
            np.arange(len(sites), dtype=np.int64),
            thresholds=priors.threshold,
            # A flat fill prior (ADVICE2 section 16): with the cohort allele frequency as the prior, wrong fills
            # exceeded the flat arm's on the development replays, so a fill weighs the carriers' evidence alone.
            cohort_alt=np.full(len(sites), 0.5),
            coordinate_sweeps=config.coordinate_sweeps,
            min_supporters=config.min_supporters,
            reverse=reverse,
        )
        del labels, likelihoods, observed
        rows, columns = np.nonzero(refit_calls != panel.released)
        filled = panel.released[rows, columns] < 0
        released = refit_calls[rows, columns] >= 0
        index = cell_class[rows, columns].astype(np.int64)
        valid = index >= 0
        np.add.at(changed_by_class, index[valid & ~filled & released], 1)
        np.add.at(filled_by_class, index[valid & filled], 1)
        guard_sites += int(np.count_nonzero(guard))
        certificate += residual.sum(axis=1)
        inference = np.where(panel.inference >= 0, refit_calls, -1).astype(np.int8)
        components.append(
            ComponentRefit(
                released=refit_calls,
                inference=inference,
                panel=replace(panel, released=refit_calls, inference=inference),
                changed_rows=rows.astype(np.int64),
                changed_sites=columns.astype(np.int64),
                gain_nats=gain[rows, columns],
                filled=filled,
            )
        )
    excluded = tuple(
        sample for sample, flag in zip(evidence.sample_ids, evidence.founder_excluded, strict=True) if flag
    )
    return FounderRefit(
        contig=painting.contig,
        sample_ids=tuple(painting.sample_ids),
        components=tuple(components),
        excluded_samples=excluded,
        class_cells=np.asarray(priors.cells, dtype=np.int64),
        class_positive=np.asarray(priors.positive, dtype=np.int64),
        class_discordance=np.asarray(priors.coefficient, dtype=np.float64),
        class_penalty=np.asarray(priors.penalty, dtype=np.float64),
        class_bound=np.asarray(priors.bound, dtype=bool),
        class_changed=changed_by_class,
        class_filled=filled_by_class,
        guard_sites=guard_sites,
        eligible_remaining=int(certificate[0]),
        refused_by_support=int(certificate[1]),
        eligible_after_fills=int(certificate[2]),
    )


@dataclass(frozen=True)
class RefitSummary:
    """Exact per-contig counts and median accepted/ambiguous-cell gain."""

    contig: str = record_field("report", dtype="str", units="1")
    components: int = record_field("report", dtype="int64", units="1")
    founder_rows: int = record_field("report", dtype="int64", units="1")
    sites: int = record_field("report", dtype="int64", units="1")
    called_alleles: int = record_field("report", dtype="int64", units="1")
    changed_alleles: int = record_field("report", dtype="int64", units="1")
    ambiguous_alleles: int = record_field("report", dtype="int64", units="1")
    filled_alleles: int = record_field("report", dtype="int64", units="1")
    changed_inference_alleles: int = record_field("report", dtype="int64", units="1")
    changed_sites: int = record_field("report", dtype="int64", units="1")
    median_changed_gain_nats: float | None = record_field("report", dtype="float64", units="nats")
    excluded_samples: tuple[str, ...] = record_field("report", dtype="tuple[str]", units="1")
    guard_sites: int = record_field("report", dtype="int64", units="1", default=0)


def summary(refit: FounderRefit) -> RefitSummary:
    """Small per-chromosome summary (log line, stage summary table)."""
    gains = np.concatenate([component.gain_nats for component in refit.components]) if refit.components else np.zeros(0)
    return RefitSummary(
        contig=refit.contig,
        components=len(refit.components),
        founder_rows=sum(c.released.shape[0] for c in refit.components),
        sites=sum(c.released.shape[1] for c in refit.components),
        called_alleles=sum(int(np.sum(c.released >= 0)) for c in refit.components),
        changed_alleles=len(gains),
        ambiguous_alleles=sum(int(np.sum(c.released[c.changed_rows, c.changed_sites] < 0)) for c in refit.components),
        filled_alleles=sum(int(np.sum(c.filled)) for c in refit.components),
        changed_inference_alleles=sum(
            int(np.sum(c.inference[c.changed_rows, c.changed_sites] >= 0)) for c in refit.components
        ),
        changed_sites=sum(len(np.unique(c.changed_sites)) for c in refit.components),
        median_changed_gain_nats=float(np.median(gains)) if len(gains) else None,
        excluded_samples=tuple(refit.excluded_samples),
        guard_sites=int(refit.guard_sites),
    )
