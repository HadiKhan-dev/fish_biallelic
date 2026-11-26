"""Per-library contamination (4.22 QC A): sufficient statistics per contig, the run's decision, corrected evidence.

Owns stages "contamination_cells" (per contig, after its painting) and "contamination" (run-wide, after every analysed
contig's cells, before the repaint). Order of the run: painting -> contamination_cells -> contamination -> repaint ->
pedigree -> family phase. The founder refit keeps the uncorrected evidence (known-truth runs: contamination left the
founders unchanged; it damaged the painting the pedigree reads and the pedigree).

- contamination_cells(c): the contig's painted genotypes (the painting product), the evidence counts and read model ->
  each library's population-law cells (g, q bin), every ordered pair's donor-law cells (g_i, g_D), the cohort's
  per-site allele-1 fraction sums, and the read model's heterozygote allele-1 mean per library, all in the contig's
  canonical search coding.
- contamination: the cells summed over the analysed contigs; per library the population-law alpha and own error, the
  flag (science.painting.contamination.decide), and for flagged libraries the best sequenced donor.
- corrected_evidence(...): a consumer's ContigEvidence with the flagged libraries' likelihood rows replaced by the
  contamination model (donor law when a donor is named, else the population plug-in); unchanged when none is flagged.

Identities: cells bind the painting's and the evidence's; the decision binds every analysed contig's cells. Consumers
(repaint, pedigree preparation, family phase) bind the decision's identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...contracts.evidence import ContigEvidence
from ...contracts.likelihoods import gather_likelihoods
from ...core.concurrency import threaded_each
from ...core.errors import InvariantError
from ...science.painting import contamination as science
from ...science.painting.parameters import (
    CONTAMINATION_ALPHA_FLOOR,
    CONTAMINATION_CARE_ALPHA,
    CONTAMINATION_DONOR_RESIDUAL,
    CONTAMINATION_Z_FLOOR,
)
from ...store.protocol import claim, publish, resume
from ...store.store import CheckpointStore
from ..framework import RecordKey, ResourceRequest, Task
from .contamination_records import CELLS, CONTAMINATION, SCHEMA, decision, expected

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ..context import StageRun

# Donor rows of the pair products handled together (memory: block x N x 9 cells x 2, plus the indicator rows).
_DONOR_BLOCK = 64

__all__ = ["cells_run", "corrected_evidence", "run", "task"]


def _read_model(decisions, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """(e, h) per library from the evidence read model actually applied; a fixed model gives e = its error and an
    even heterozygote."""
    model = decisions.read_model
    fit = model.candidates.get(model.selected) if model.applied and isinstance(model.candidates, Mapping) else None
    if fit is None:
        return np.full(n_samples, np.nan), np.full(n_samples, 0.5)
    error, heterozygote = fit.error_and_heterozygote()
    return np.asarray(error, dtype=np.float64), np.asarray(heterozygote, dtype=np.float64)


def cells_run(root: str, contig: str, resolved: ResolvedConfiguration, *, team: int = 1) -> str:
    """Publish the contig's contamination cells."""
    del resolved
    from ..evidence import records as evidence_records
    from ..inputs.products import required_product, task_inputs
    from . import records as painting_records

    reader, store = task_inputs(root)
    file = store.contig(contig)
    request = expected(store, reader.manifest, product="cells", contig=contig)
    with claim(file, CELLS, request.identity, blocking=False) as needed:
        if needed:
            painting = required_product(reader, painting_records, "painting", contig)
            evidence_request = evidence_records.expected(store, reader.manifest, product="evidence", contig=contig)
            # counts and positions only (the likelihoods are not read here); rows follow the manifest's sample axis
            arrays = evidence_records.load_arrays(file, ("counts", "positions"),
                                                  expected_identity=evidence_request.identity)
            decisions = reader.read(reader.request(evidence_records, "summary", contig), contig)
            if decisions is None:
                raise InvariantError(f"{contig}: contamination cells require the evidence summary")
            reflect = evidence_records.load_search_alleles_reversed(file, expected_identity=evidence_request.identity)
            counts = np.asarray(arrays["counts"])
            alt = counts[:, :, 0 if reflect else 1].astype(np.int64)
            depth = counts.sum(axis=2, dtype=np.int64)
            del counts
            genotypes = science.painted_genotypes(
                painting, np.asarray(arrays["positions"]), search_alleles_reversed=reflect,
            )
            sample_ids = tuple(str(x) for x in reader.manifest["sample_ids"])
            if len(sample_ids) != alt.shape[0]:
                raise InvariantError(f"{contig}: evidence rows differ from the manifest's sample axis")
            fraction_sum, observed_count = science.cohort_sums(alt, depth)
            n = alt.shape[0]

            def population(i: int):
                q = science.leave_one_out_frequency(alt[i], depth[i], fraction_sum, observed_count)
                return science.population_cells(alt[i], depth[i], genotypes[i], q)

            cells = threaded_each(population, range(n))
            # one block of library rows per thread of the task's team (at most _DONOR_BLOCK rows), sharing the
            # genotype indicators built once for the contig
            width = max(1, min(_DONOR_BLOCK, -(-n // max(1, int(team)))))
            blocks = [(start, min(n, start + width)) for start in range(0, n, width)]
            indicators = science.genotype_indicators(genotypes)
            donor_alt = np.empty((n, n, 3, 3))
            donor_ref = np.empty((n, n, 3, 3))

            def donor_block(bounds):
                lo, hi = bounds
                block_alt, block_ref = science.donor_cell_rows(alt, depth, genotypes, lo, hi, indicators)
                donor_alt[lo:hi], donor_ref[lo:hi] = block_alt, block_ref

            threaded_each(donor_block, blocks)
            _, heterozygote = _read_model(decisions, n)
            if reflect:
                heterozygote = 1.0 - heterozygote
            publish(file, CELLS, request.identity, dict(
                schema=SCHEMA,
                sample_ids=list(sample_ids),
                population_alt=np.stack([c[0] for c in cells]),
                population_ref=np.stack([c[1] for c in cells]),
                donor_alt=donor_alt,
                donor_ref=donor_ref,
                heterozygote=heterozygote,
                painted_sites=(genotypes >= 0).sum(axis=1).astype(np.int64),
                fraction_sum=fraction_sum,
                observed_count=observed_count.astype(np.int32),
            ))
    return request.identity


def run(root: str, resolved: ResolvedConfiguration, *, team: int = 1) -> str:
    """Publish the run's contamination decision from every analysed contig's cells."""
    del resolved, team
    from ..inputs.products import task_inputs

    reader, store = task_inputs(root)
    manifest = reader.manifest
    request = expected(store, manifest, product="contamination")
    with claim(store.run, CONTAMINATION, request.identity, blocking=False) as needed:
        if needed:
            totals: dict[str, Any] = {}
            heterozygotes, sites = [], []
            for contig in AnalysisContigs.from_manifest(manifest).included:
                part = expected(store, manifest, product="cells", contig=contig)
                saved = resume(store.contig(contig), CELLS, part.identity)
                if saved is None:
                    raise InvariantError(
                        f"the contamination decision requires every analysed contig's cells ({contig})"
                    )
                payload = saved.payload
                for name in ("population_alt", "population_ref", "donor_alt", "donor_ref"):
                    value = np.asarray(payload[name], dtype=np.float64)
                    totals[name] = value if name not in totals else totals[name] + value
                heterozygotes.append(np.asarray(payload["heterozygote"], dtype=np.float64))
                sites.append(np.asarray(payload["painted_sites"], dtype=np.float64))
                sample_ids = tuple(payload["sample_ids"])
            weights = np.maximum(np.stack(sites), 1.0)
            heterozygote = (np.stack(heterozygotes) * weights).sum(axis=0) / weights.sum(axis=0)
            n = len(heterozygote)
            population = threaded_each(
                lambda i: science.fit_population(totals["population_alt"][i], totals["population_ref"][i],
                                                 float(heterozygote[i])),
                range(n),
            )
            alphas = np.array([fit.alpha for fit in population])
            z = science.robust_z(alphas)
            flagged = np.flatnonzero((alphas >= CONTAMINATION_ALPHA_FLOOR) & (z >= CONTAMINATION_Z_FLOOR))
            errors = np.array([fit.error for fit in population])

            def scan(i: int):
                # rank donors by the likelihood-ratio statistic on each pair's own shared sites (log likelihoods over
                # different site sets are not comparable), all candidates at once; refine the best one
                candidates = np.flatnonzero(np.arange(n) != i)
                ranking = science.donor_statistics(totals["donor_alt"][i, candidates],
                                                   totals["donor_ref"][i, candidates], float(heterozygote[i]),
                                                   errors[candidates])
                j = int(candidates[int(np.argmax(ranking))])  # first maximum: the smaller index on ties
                best = science.fit_donor(totals["donor_alt"][i, j], totals["donor_ref"][i, j],
                                         float(heterozygote[i]), float(errors[j]))
                same = science.same_homozygote_fraction(totals["donor_alt"][i, j], totals["donor_ref"][i, j])
                return j, best, same

            scanned = dict(zip(flagged.tolist(), threaded_each(scan, flagged.tolist()), strict=True))
            # the unflagged libraries' own same-homozygote minor fraction, pooled over every partner
            unflagged = np.setdiff1d(np.arange(n), flagged)
            pooled_alt = totals["donor_alt"].sum(axis=1)
            pooled_ref = totals["donor_ref"].sum(axis=1)
            own = np.array([science.same_homozygote_fraction(pooled_alt[i], pooled_ref[i]) for i in unflagged])
            baseline = float(np.nanmedian(own)) if np.isfinite(own).any() else 0.0
            outcome = science.decide(population, [scanned.get(i) for i in range(n)],
                                     alpha_floor=CONTAMINATION_ALPHA_FLOOR, z_floor=CONTAMINATION_Z_FLOOR,
                                     donor_residual=CONTAMINATION_DONOR_RESIDUAL, baseline=baseline)
            correction_alpha = np.where(outcome.donor >= 0, outcome.donor_alpha, outcome.alpha)
            # with no named donor, an own error at the grid's top means the fit absorbed foreign reads as error: alpha
            # is then a lower bound (one unsequenced source at 20 % read 0.037) and the library is marked for care
            at_bound = outcome.flagged & (outcome.donor < 0) & (outcome.error >= science.ERROR_GRID[-1] * (1 - 1e-12))
            publish(store.run, CONTAMINATION, request.identity, dict(
                schema=SCHEMA,
                sample_ids=list(sample_ids),
                alpha=outcome.alpha,
                error=outcome.error,
                statistic=outcome.statistic,
                z=outcome.z,
                flagged=outcome.flagged,
                donor=outcome.donor,
                donor_alpha=outcome.donor_alpha,
                donor_error=outcome.donor_error,
                donor_statistic=outcome.donor_statistic,
                residual=outcome.residual,
                same_homozygote_baseline=baseline,
                heterozygote=heterozygote,
                alpha_lower_bound=at_bound,
                interpret_with_care=outcome.flagged & ((correction_alpha > CONTAMINATION_CARE_ALPHA) | at_bound),
            ))
    return request.identity


def corrected_evidence(store: CheckpointStore, manifest: Mapping[str, Any], contig: str, evidence: ContigEvidence,
                       *, evidence_identity: str | None = None) -> ContigEvidence:
    """Replace flagged likelihood rows, or return the original evidence when there are no flags."""
    corrected, _ = corrected_evidence_with_rows(
        store, manifest, contig, evidence, evidence_identity=evidence_identity,
    )
    return corrected


def corrected_evidence_with_rows(
    store: CheckpointStore, manifest: Mapping[str, Any], contig: str, evidence: ContigEvidence,
    *, evidence_identity: str | None = None,
) -> tuple[ContigEvidence, tuple[int, ...]]:
    """Return corrected evidence and the sample indices whose likelihoods were replaced.

    The donor law uses the donor's painted genotype, or its likelihood row where unpainted.
    Derive the evidence record identity here when correction needs it and the caller lacks it.
    """
    from ..evidence import records as evidence_records
    from . import records as painting_records

    _, product = decision(store, manifest)
    flagged = np.flatnonzero(np.asarray(product["flagged"], dtype=bool))
    if flagged.size == 0:
        return evidence, ()
    if tuple(product["sample_ids"]) != tuple(evidence.sample_ids):
        raise InvariantError("contamination decision and evidence sample axes differ")
    if evidence_identity is None:
        evidence_identity = evidence_records.expected(store, manifest, product="evidence", contig=contig).identity
    file = store.contig(contig)
    cells = resume(file, CELLS, expected(store, manifest, product="cells", contig=contig).identity)
    if cells is None:
        raise InvariantError(f"{contig}: contamination correction requires the contig's cells")
    fraction_sum = np.asarray(cells.payload["fraction_sum"], dtype=np.float64)
    observed_count = np.asarray(cells.payload["observed_count"], dtype=np.int64)
    donor = np.asarray(product["donor"], dtype=np.int64)
    reflect = evidence.search_alleles_reversed
    genotype_step = -1 if reflect else 1
    donors = sorted({int(donor[i]) for i in flagged if donor[i] >= 0})
    painted = None
    if donors:
        fine = painting_records.expected(store, manifest, product="painting", contig=contig)
        painting = painting_records.load(file, fine.identity, emissions=False)
        if painting is None:
            raise InvariantError(f"{contig}: the donor law requires the contig's painting")
        painted = science.painted_genotypes(
            painting, np.asarray(evidence.positions), rows=donors, search_alleles_reversed=reflect,
        )
    likelihoods = np.array(gather_likelihoods(evidence.likelihoods, slice(None)), dtype=np.float64)
    all_counts = np.asarray(
        evidence_records.load_arrays(file, ("counts",), expected_identity=evidence_identity)["counts"]
    )
    for i in flagged:
        counts = all_counts[i]
        alt = counts[:, 0 if reflect else 1].astype(np.int64)
        depth = counts.sum(axis=1, dtype=np.int64)
        q = science.leave_one_out_frequency(alt, depth, fraction_sum, observed_count)
        if donor[i] >= 0:
            j = int(donor[i])
            q = science.donor_frequency(
                painted[donors.index(j)], float(product["error"][j]), likelihoods[j, :, ::genotype_step], q,
            )
            alpha, error = float(product["donor_alpha"][i]), float(product["donor_error"][i])
        else:
            alpha, error = float(product["alpha"][i]), float(product["error"][i])
        corrected = science.corrected_likelihoods(alt, depth, error, float(product["heterozygote"][i]), alpha, q)
        likelihoods[i] = corrected[:, ::genotype_step]
    return replace(evidence, likelihoods=likelihoods), tuple(flagged.tolist())


def cells_task(key, context: StageRun) -> Task:
    """A contig's contamination cells."""
    return Task(
        key, cells_run, (context.root, key.scope, context.resolved),
        outputs=(RecordKey(key.scope, CELLS),),
        resources=ResourceRequest(1, context.cores), thread_argument="team",
    )


def task(key, context: StageRun) -> Task:
    """Declare the contamination cells (per contig) or the run's decision (the stage of the key)."""
    if key.stage == "contamination_cells":
        return cells_task(key, context)
    return Task(
        key, run, (context.root, context.resolved),
        outputs=(RecordKey("run", CONTAMINATION),),
        resources=ResourceRequest(1, context.cores), thread_argument="team",
    )
