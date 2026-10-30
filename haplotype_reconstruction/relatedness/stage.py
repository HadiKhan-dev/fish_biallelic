"""The relatedness stage: pairwise IBD and full-sib families from the evidence records alone (genotype likelihoods),
before any genome-wide stage, so later stages can use them without a feedback loop."""
from __future__ import annotations

import time

import numpy as np

from .. import execution
from ..evidence import records as evidence_records
from . import families as family_model
from . import records
from .config import RelatednessConfig
from .kernels import IBD_MASKS, allele_frequency, family_terms, pair_ibd


def _contig_terms(store, contig, sample_ids, config: RelatednessConfig, genetic_map, pairs, posterior, loglik):
    """One contig: the family model's site terms (returned) and the pairwise IBD chain (accumulated in place)."""
    file = store.contig(contig)
    recorded = tuple(map(str, file.get(evidence_records.EVIDENCE, items=("sample_ids",))["sample_ids"]))
    if recorded != tuple(sample_ids):
        raise ValueError(f"{contig}: the evidence record's sample axis differs from the run's")
    arrays = evidence_records.load_arrays(file, ("likelihoods", "observed", "positions"))
    gl = np.asarray(arrays["likelihoods"], dtype=np.float64)
    observed = np.asarray(arrays["observed"], dtype=np.bool_)
    positions = np.asarray(arrays["positions"])
    p = allele_frequency(gl, observed)
    keep = (np.minimum(p, 1 - p) >= config.minor_allele_frequency) & (observed.mean(axis=0) >= config.minimum_observed)
    sites = np.flatnonzero(keep)[::config.family_site_step]
    terms = family_terms(gl[:, sites], observed[:, sites], p[sites])
    # Release 3.2: support units follow the measured map: a supplied map over the span its knots cover (the fallback
    # rate, and a map's extrapolated tails, measure nothing), or the run's map everywhere when the rate is stated as
    # the organism's genome-wide average ([genetic_map] rate_is_measured).
    units = support_units(_measured_coordinates(genetic_map, positions[sites], config.rate_is_measured),
                          config.support_unit_morgans)
    step = config.ibd_site_step
    if not len(positions):
        return terms, units
    morgans = genetic_map.cumulative_morgans(positions[::step])
    stay = np.ones(len(morgans))
    stay[1:] = np.exp(-config.ibd_switch_rate * np.diff(morgans))
    pair_ibd(np.ascontiguousarray(gl[:, ::step]), np.ascontiguousarray(observed[:, ::step]),
             np.ascontiguousarray(p[::step]), stay, pairs, IBD_MASKS, posterior, loglik)
    return terms, units


def _measured_coordinates(genetic_map, positions, rate_is_measured: bool) -> np.ndarray:
    """float64 [n]: map coordinates (Morgans) of ``positions`` (bp) as far as a measurement supports them: the run's
    map with ``rate_is_measured``, else a supplied map with the positions clipped to its knots, else zeros (the
    pedigree's exposure gate measures lengths the same way: pedigree.prepare.measured_coordinates)."""
    positions = np.asarray(positions, dtype=np.float64)
    if rate_is_measured:
        return np.asarray(genetic_map.cumulative_morgans(positions), dtype=np.float64)
    if not genetic_map.has_map:
        return np.zeros(len(positions))
    knots = genetic_map.positions_bp
    return np.asarray(genetic_map.cumulative_morgans(np.clip(positions, knots[0], knots[-1])), dtype=np.float64)


def support_units(morgans: np.ndarray, unit_morgans: float) -> np.ndarray:
    """int64 [k + 1]: boundaries (site offsets) of a chromosome's support units: max(1, floor(length / unit))
    consecutive pieces of equal measured map length over its family sites (``morgans``: their measured map
    coordinates, nondecreasing; one piece without a measured length)."""
    n_sites = len(morgans)
    if n_sites == 0:
        return np.asarray([0, 0], dtype=np.int64)
    length = float(morgans[-1] - morgans[0])
    pieces = max(1, int(np.floor(length / unit_morgans)))
    cuts = morgans[0] + length * np.arange(1, pieces) / pieces
    return np.concatenate(([0], np.searchsorted(morgans, cuts, side="left"), [n_sites])).astype(np.int64)


def run(store, contigs: tuple[str, ...], sample_ids: tuple[str, ...], config: RelatednessConfig, genetic_maps,
        budget) -> records.RelatednessProduct:
    """Estimate pairwise IBD and full-sib families; returns (and stores) the run record ``relatedness``.

    Reads each contig's evidence record; uses no painting, pedigree, phase or cohort label. ``genetic_maps``
    (GeneticMapSet) sets the IBD chain's switching distance; ``budget`` the Numba team.
    """
    contigs, sample_ids = tuple(contigs), tuple(sample_ids)
    store.require(evidence_records.EVIDENCE, contigs)
    store.bind_identity(store.run, records.PRODUCT, records.product_identity(store, contigs, config, genetic_maps))
    if store.run.has(records.PRODUCT):
        return records.load_product(store)
    product = compute(store, contigs, sample_ids, config, genetic_maps, budget)
    store.run.put(records.PRODUCT, records.encode_product(product))
    store.mark_stage_complete(records.STAGE)
    return product


def compute(store, contigs, sample_ids, config: RelatednessConfig, genetic_maps, budget) -> records.RelatednessProduct:
    """The product from the evidence records (reads only; ``run`` binds and stores it)."""
    contigs, sample_ids = tuple(contigs), tuple(sample_ids)
    started = time.monotonic()
    n = len(sample_ids)
    first, second = np.triu_indices(n, k=1)
    pairs = np.ascontiguousarray(np.stack([first, second], axis=1).astype(np.int64))
    posterior = np.zeros((len(pairs), 4))
    loglik = np.zeros((len(pairs), len(IBD_MASKS)))
    child_terms, log_prior, log_posterior, ranges, chromosome_sites = [], [], [], [], []
    with execution.numba_threads(budget.threads()):
        for contig in contigs:
            terms, units = _contig_terms(store, contig, sample_ids, config, genetic_maps.for_contig(contig),
                                         pairs, posterior, loglik)
            lo = sum(sites.shape[1] for sites in child_terms)
            chromosome_sites.append(int(terms[0].shape[1]))
            ranges.extend((lo + int(a), lo + int(b)) for a, b in zip(units[:-1], units[1:]) if b > a)
            for collected, value in zip((child_terms, log_prior, log_posterior), terms):
                collected.append(value)
        C = np.ascontiguousarray(np.concatenate(child_terms, axis=1))
        logpi = np.ascontiguousarray(np.concatenate(log_prior, axis=0))
        logpost = np.ascontiguousarray(np.concatenate(log_posterior, axis=1))
        del child_terms, log_prior, log_posterior
        print(f"RELATEDNESS: {n} samples, {C.shape[1]} family sites over {len(contigs)} chromosomes "
              f"({len(ranges)} support units), {len(pairs)} pairs ({time.monotonic() - started:.0f} s)", flush=True)
        fit = family_model.fit(C, logpi, logpost, ranges, max_sweeps=config.max_sweeps,
                               log=lambda line: print(line, flush=True))
    family = np.full(n, -1, dtype=np.int64)
    for index, members in enumerate(fit.members):
        family[members] = index
    product = records.RelatednessProduct(
        sample_ids=sample_ids, contigs=contigs, pairs=pairs,
        ibd=posterior[:, :3] / np.maximum(posterior[:, 3:4], 1.0), ibd_sites=posterior[:, 3].copy(),
        lr_ibd0=loglik[:, 0] - loglik[:, 1], lr_ibd2=loglik[:, 0] - loglik[:, 2], family=family,
        parents=fit.parents, member_support=fit.member_support, parent_support=fit.parent_support,
        half_sib=fit.half_sib, half_sib_gain=fit.half_sib_gain, loglik=fit.loglik, sweeps=fit.sweeps,
        converged=fit.converged, family_sites=int(C.shape[1]),
        chromosome_sites=tuple(chromosome_sites), support_units=len(ranges))
    sizes = np.bincount(family[family >= 0])
    print(f"RELATEDNESS: {len(fit.members)} families (largest {int(sizes.max())}), "
          f"{int(np.count_nonzero(fit.parents >= 0))} sequenced parent slots, {len(fit.half_sib)} half-sib links, "
          f"{int(np.count_nonzero(fit.member_support >= 0.75))} of {n} samples on their family on >= 75% of "
          f"support units; "
          f"{'converged' if fit.converged else 'sweep limit reached'} after {fit.sweeps} sweeps "
          f"({time.monotonic() - started:.0f} s)", flush=True)
    return product
