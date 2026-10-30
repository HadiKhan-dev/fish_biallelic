"""OUTPUT/relatedness/: the relatedness stage's families, pairwise IBD summaries and half-sib links.

families.csv: one row per sample (its family, the share of chromosomes on which that family is its best placement,
the family's two parents: a sequenced sample or empty for a latent parent, with the parent's chromosome support).
pairs.csv: one row per sample pair (posterior IBD0/IBD1/IBD2 shares of jointly observed sites, their count, and the
IBD0 and IBD2 stretch evidence). half_sib_links.csv: latent families better explained as sharing one latent parent.
summary.json: the fit. These are inferences from genotype likelihoods alone, not pedigree truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tables import write_frame, write_json


def write_relatedness(output, product, input_ids=None) -> list[Path]:
    """Write the product's tables; per-sample rows in the run's input order (``input_ids``)."""
    from ..evidence.sample_order import frame_in_input_order
    directory = Path(output) / "relatedness"
    ids = np.asarray(product.sample_ids, dtype=object)
    parent = lambda index: ids[index] if index >= 0 else ""   # noqa: E731
    families = pd.DataFrame(dict(
        Sample=ids, Family=product.family, FamilySupport=product.member_support,
        Parent1=[parent(product.parents[f, 0]) for f in product.family],
        Parent1Support=[product.parent_support[f, 0] for f in product.family],
        Parent2=[parent(product.parents[f, 1]) for f in product.family],
        Parent2Support=[product.parent_support[f, 1] for f in product.family]))
    pairs = pd.DataFrame(dict(
        SampleA=ids[product.pairs[:, 0]], SampleB=ids[product.pairs[:, 1]], IBD0=product.ibd[:, 0],
        IBD1=product.ibd[:, 1], IBD2=product.ibd[:, 2], Sites=product.ibd_sites, IBD0Evidence=product.lr_ibd0,
        IBD2Evidence=product.lr_ibd2))
    links = pd.DataFrame(dict(FamilyA=product.half_sib[:, 0], FamilyB=product.half_sib[:, 1],
                              Gain=product.half_sib_gain))
    written = [write_frame(directory / "families.csv", frame_in_input_order(families, input_ids)),
               write_frame(directory / "pairs.csv", pairs), write_frame(directory / "half_sib_links.csv", links)]
    written.append(write_json(directory / "summary.json", dict(
        families=int(len(product.parents)), sequenced_parent_slots=int(np.count_nonzero(product.parents >= 0)),
        half_sib_links=int(len(product.half_sib)), loglik=float(product.loglik), sweeps=int(product.sweeps),
        converged=bool(product.converged), family_sites=int(product.family_sites),
        chromosome_sites=dict(zip(product.contigs, product.chromosome_sites)))))
    print(f"Relatedness output: {directory / 'families.csv'}", flush=True)
    return written
