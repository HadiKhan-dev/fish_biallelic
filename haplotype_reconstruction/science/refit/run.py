"""Refit the painted founder panels without changing their released painting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ...contracts.likelihoods import gather_likelihoods
from ...contracts.refit import ComponentRefit, FounderRefit
from ...contracts.roles import record_field
from ...core.errors import InvariantError
from .parameters import RefitParameters
from .refit import refit_alleles

if TYPE_CHECKING:
    from ...contracts.evidence import ContigEvidence
    from ...contracts.painting import Painting


def refit_chromosome(
    painting: Painting, evidence: ContigEvidence, config: RefitParameters, *, reverse: bool = False
) -> FounderRefit:
    """Refit every painted component of one chromosome; the painting is unchanged.

    ``painting`` must carry its panels (painting.records.load(..., assembly=assembly));
    ``evidence`` is the contig's evidence (its founder view is used here).
    """
    founder = evidence.founder_view()
    components = []
    for component in painting.components:
        panel = component.panel
        if panel is None:
            raise InvariantError("refit requires the painted component panel")
        sites = panel.site_indices()
        labels = component.site_labels(panel.positions)
        likelihoods, observed, kernel_sites = founder.likelihoods, founder.observed, sites
        if not isinstance(likelihoods, np.ndarray) or not isinstance(observed, np.ndarray):
            likelihoods = gather_likelihoods(likelihoods, sites)
            observed = np.ascontiguousarray(observed[:, sites])
            kernel_sites = np.arange(len(sites), dtype=np.int64)
        refit_calls, gain = refit_alleles(
            panel.released,
            labels[:, 0],
            labels[:, 1],
            likelihoods,
            observed,
            kernel_sites,
            robustness_epsilon=painting.parameters.robustness_epsilon,
            minimum_gain_nats=config.minimum_gain_nats,
            coordinate_sweeps=config.coordinate_sweeps,
            fill_min_supporters=config.fill_min_supporters,
            reverse=reverse,
        )
        del labels
        rows, columns = np.nonzero(refit_calls != panel.released)
        inference = np.where(panel.inference >= 0, refit_calls, -1).astype(np.int8)
        components.append(
            ComponentRefit(
                released=refit_calls,
                inference=inference,
                panel=replace(panel, released=refit_calls, inference=inference),
                changed_rows=rows.astype(np.int64),
                changed_sites=columns.astype(np.int64),
                gain_nats=gain[rows, columns],
                filled=panel.released[rows, columns] < 0,
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
    )


@dataclass(frozen=True)
class RefitSummary:
    """Exact per-contig counts and median accepted/ambiguous-cell gain (P0 FR06)."""

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
    )
