"""OUTPUT/relatedness/: the relatedness stage's families, pairwise IBD summaries and half-sib links.

families.csv: one row per sample (its family, the share of chromosomes on which that family is its best placement,
the family's two parents: a sequenced sample or empty for a latent parent, with the parent's chromosome support).
pairs.csv: one row per sample pair (posterior IBD0/IBD1/IBD2 shares of jointly observed sites, their count, and the
IBD0 and IBD2 stretch evidence). half_sib_links.csv: latent families better explained as sharing one latent parent.
summary.json: the fit (with the family search kept and its margin). These are inferences from genotype likelihoods
alone, not pedigree truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...results.reader import RelatednessView

import numpy as np
import pandas as pd

from .tables import write_frame, write_json


def write_relatedness(
    output: str | Path, product: RelatednessView, input_ids: Sequence[str] | None = None
) -> list[Path]:
    """Write the product's tables; per-sample rows in the run's input order (``input_ids``)."""
    from .tables import frame_in_input_order

    directory = Path(output) / "relatedness"
    ids = np.asarray(product.sample_ids, dtype=object)
    parent = lambda index: ids[index] if index >= 0 else ""  # noqa: E731
    families = pd.DataFrame(
        dict(
            Sample=ids,
            Family=product.family,
            FamilySupport=product.member_support,
            Parent1=[parent(product.parents[f, 0]) for f in product.family],
            Parent1Support=[product.parent_support[f, 0] for f in product.family],
            Parent2=[parent(product.parents[f, 1]) for f in product.family],
            Parent2Support=[product.parent_support[f, 1] for f in product.family],
        )
    )
    links = pd.DataFrame(
        dict(FamilyA=product.half_sib[:, 0], FamilyB=product.half_sib[:, 1], Gain=product.half_sib_gain)
    )
    written = [write_frame(directory / "families.csv", frame_in_input_order(families, input_ids))]
    if product.pairs is not None:
        ibd = product.ibd
        assert ibd is not None  # pairs and their IBD shares come from one recorded pair product
        pairs = pd.DataFrame(
            dict(
                SampleA=ids[product.pairs[:, 0]],
                SampleB=ids[product.pairs[:, 1]],
                IBD0=ibd[:, 0],
                IBD1=ibd[:, 1],
                IBD2=ibd[:, 2],
                Sites=product.ibd_sites,
                IBD0Evidence=product.lr_ibd0,
                IBD2Evidence=product.lr_ibd2,
            )
        )
        written.append(write_frame(directory / "pairs.csv", pairs))
    else:
        (directory / "pairs.csv").unlink(missing_ok=True)
    written.append(write_frame(directory / "half_sib_links.csv", links))
    written.append(
        write_json(
            directory / "summary.json",
            dict(
                families=int(len(product.parents)),
                sequenced_parent_slots=int(np.count_nonzero(product.parents >= 0)),
                half_sib_links=int(len(product.half_sib)),
                loglik=float(product.loglik),
                sweeps=int(product.sweeps),
                converged=bool(product.converged),
                search=str(product.search),
                search_margin=float(product.search_margin) if np.isfinite(product.search_margin) else None,
                family_sites=int(product.family_sites),
                chromosome_sites=dict(zip(product.contigs, product.chromosome_sites, strict=False)),
            ),
        )
    )
    return written


# Published column schemas: B reports and P0 outputs inventory D6 / OU15.
# Row-dict tables retain B's headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
FAMILY_COLUMNS = ("Sample", "Family", "FamilySupport", "Parent1", "Parent1Support", "Parent2", "Parent2Support")
"""Columns in published order.

    Sample: sample identifier (text).
    Family: canonical dense family rank, ordered by smallest member (1).
    FamilySupport: fraction of chromosomes supporting this sample-family assignment (1).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent1Support: fraction of chromosomes supporting the first parent slot (1).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    Parent2Support: fraction of chromosomes supporting the second parent slot (1).
"""

PAIR_COLUMNS = ("SampleA", "SampleB", "IBD0", "IBD1", "IBD2", "Sites", "IBD0Evidence", "IBD2Evidence")
"""Columns in published order.

    SampleA: first sample identifier (text).
    SampleB: second sample identifier (text).
    IBD0: posterior share at jointly observed sites with zero alleles identical by descent (1).
    IBD1: posterior share at jointly observed sites with one allele identical by descent (1).
    IBD2: posterior share at jointly observed sites with two alleles identical by descent (1).
    Sites: sites (1).
    IBD0Evidence: recorded IBD0 stretch evidence (nats).
    IBD2Evidence: recorded IBD2 stretch evidence (nats).
"""

HALF_SIB_COLUMNS = ("FamilyA", "FamilyB", "Gain")
"""Columns in published order.

    FamilyA: first canonical dense family rank (1).
    FamilyB: second canonical dense family rank (1).
    Gain: half-sibling linkage log-likelihood gain (nats).
"""
