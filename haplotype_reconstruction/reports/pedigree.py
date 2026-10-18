"""OUTPUT/pedigree/: the pedigree stage's report tables, its evidence calibration and a run summary.

One CSV per table of the product's PedigreeReport (tier_b_relationships.csv is the primary call table; the others
are the Tier-A and complete views, partial relationships, candidate sets, explanations and diagnostics),
evidence_calibration.json (the cross-chromosome evidence scaling) and summary.json (the scored and omitted
chromosomes, what Tier A/B can release with them, the final family solve, the full-marker exclusion and the
eligibility design: why a tier may release nothing). Real-data calls are inferences conditional on the paintings
and the stated eligibility design, not pedigree truth.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

from .tables import write_frame, write_json


def write_pedigree(output, product, input_ids=None) -> list[Path]:
    """Write every table of ``product.report``, the calibration and the run summary; returns the paths written.
    ``input_ids``: the run's input sample order (the store manifest's, evidence.sample_order.store_input_order);
    per-sample tables list their rows in it (the records keep the canonical order, release 2.1)."""
    from ..evidence.sample_order import frame_in_input_order
    directory = Path(output) / "pedigree"
    written = [write_frame(directory / f"{field.name}.csv",
                           frame_in_input_order(getattr(product.report, field.name), input_ids))
               for field in fields(product.report)]
    written.append(write_json(directory / "evidence_calibration.json", product.calibration))
    written.append(write_json(directory / "summary.json", dict(
        informative_contigs=list(product.informative_contigs), omitted=dict(product.omitted),
        reach=asdict(product.reach), family_solve=product.family_solve, exclusion=product.exclusion,
        eligibility=product.eligibility)))
    print(f"Pedigree output: {directory / 'tier_b_relationships.csv'}", flush=True)
    return written
