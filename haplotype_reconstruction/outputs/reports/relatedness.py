"""OUTPUT/relatedness/: the relatedness stage's families and half-sib links.

families.csv: one row per sample (its family, the share of chromosomes on which that family is its best placement,
the family's two parent slots: a sequenced sample or empty for a latent slot, with the slot's support).
A slot sample's genotypes serve the model as a parent's would; it may be a parent, child (the two directions tie
for a one-member family), sib or another close relative. Parent calls, direction and their support belong to the
pedigree stage.
half_sib_links.csv: latent families better explained as sharing one latent parent.
summary.json: the fit (with the family search kept and its margin) and, under recombination_rate, the genome-wide
recombination rate estimated from the families (rl20): estimate and 95 % interval (cM/Mb), meioses, the families used
and dropped with reasons, the gate and the working-map decision, and the warning printed in the run log (null when
none). These are inferences from genotype likelihoods alone, not pedigree truth. Releases up to 4.17.0 could also
write pairs.csv, a pairwise IBD table whose model does
not hold in a cross (docs/methods/relatedness.md, "Pairwise IBD"); a stale copy is removed when the tables are
rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...contracts.relatedness import RelatednessFamilies

import numpy as np
import pandas as pd

from .tables import write_frame, write_json


def write_relatedness(
    output: str | Path, product: RelatednessFamilies, input_ids: Sequence[str] | None = None
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
            ParentSlot1=[parent(product.parents[f, 0]) for f in product.family],
            ParentSlot1Support=[product.parent_support[f, 0] for f in product.family],
            ParentSlot2=[parent(product.parents[f, 1]) for f in product.family],
            ParentSlot2Support=[product.parent_support[f, 1] for f in product.family],
        )
    )
    links = pd.DataFrame(
        dict(FamilyA=product.half_sib[:, 0], FamilyB=product.half_sib[:, 1], Gain=product.half_sib_gain)
    )
    rate = getattr(product, "rate", None)  # absent from records built outside the relatedness stage
    written = [write_frame(directory / "families.csv", frame_in_input_order(families, input_ids))]
    # Written only by earlier releases (an output run resumed under this one would otherwise keep it).
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
                recombination_rate=_rate_summary(rate) if rate is not None else None,
            ),
        )
    )
    return written


def _number(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def _rate_summary(rate) -> dict:
    """The rate decision as plain JSON; per-chromosome values are diagnostics only."""
    return dict(
        estimate_cm_per_mb=_number(rate.estimate_cm_per_mb),
        interval_cm_per_mb=[_number(rate.interval_low_cm_per_mb), _number(rate.interval_high_cm_per_mb)],
        interval_kind=rate.interval_kind,
        meioses=int(rate.meioses),
        determined_sites_per_pair=_number(rate.determined_sites),
        families_used=sum(1 for status in rate.family_status if status == "used"),
        error_rate=_number(rate.error_rate),
        artefact_rate_per_mb=_number(rate.artefact_rate_per_mb),
        artefact_length_kb=_number(rate.artefact_length_kb),
        at_grid_edge=bool(rate.at_grid_edge),
        gate_passed=bool(rate.gate_passed),
        gate_reason=rate.gate_reason,
        supplied_rate_cm_per_mb=_number(rate.supplied_rate_cm_per_mb),
        infer_rate=bool(rate.infer_rate),
        decision=rate.decision,
        decision_reason=rate.decision_reason,
        working_rate_cm_per_mb=_number(rate.working_rate_cm_per_mb),
        contigs_using_working_rate=list(rate.map_free_contigs) if rate.decision == "adopted" else [],
        warning=rate.warning or None,
        families=[
            dict(
                family=int(f),
                members=int(n),
                model=model,
                pairs=int(pairs),
                determined_sites_per_pair=_number(sites),
                estimate_cm_per_mb=_number(estimate),
                status=status,
                inconsistency_nats=_number(lr),
            )
            for f, n, model, pairs, sites, estimate, status, lr in zip(
                rate.family, rate.family_members, rate.family_model, rate.family_pairs, rate.family_sites,
                rate.family_estimate_cm_per_mb, rate.family_status, rate.family_inconsistency, strict=True,
            )
        ],
        per_chromosome_cm_per_mb_diagnostic=dict(
            zip(rate.contigs, (_number(v) for v in rate.per_chromosome_cm_per_mb), strict=True)
        ),
    )


# Published report column schemas.
# Row-dict tables use a headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
FAMILY_COLUMNS = (
    "Sample", "Family", "FamilySupport", "ParentSlot1", "ParentSlot1Support", "ParentSlot2", "ParentSlot2Support"
)
"""Columns in published order.

    Sample: sample identifier (text).
    Family: canonical dense family rank, ordered by smallest member (1).
    FamilySupport: fraction of chromosomes supporting this sample-family assignment (1).
    ParentSlot1: first slot sample whose genotypes serve the family model as a parent's would; may be a parent, child, sib or another close relative; empty for a latent slot (text).
    ParentSlot1Support: fraction of nonempty support units on which the first slot sample beats its latent replacement; parent calls, direction and their support belong to the pedigree stage (1).
    ParentSlot2: second slot sample whose genotypes serve the family model as a parent's would; may be a parent, child, sib or another close relative; empty for a latent slot (text).
    ParentSlot2Support: fraction of nonempty support units on which the second slot sample beats its latent replacement; the two directions tie for a one-member family (1).
"""

HALF_SIB_COLUMNS = ("FamilyA", "FamilyB", "Gain")
"""Columns in published order.

    FamilyA: first canonical dense family rank (1).
    FamilyB: second canonical dense family rank (1).
    Gain: half-sibling linkage log-likelihood gain (nats).
"""
