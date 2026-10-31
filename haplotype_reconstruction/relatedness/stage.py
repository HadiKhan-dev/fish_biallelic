"""The relatedness stage: pairwise IBD and full-sib families from the evidence records alone (genotype likelihoods),
before any genome-wide stage, so later stages can use them without a feedback loop."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import execution
from ..evidence import records as evidence_records
from . import families as family_model
from . import records
from .config import RelatednessConfig
from .kernels import IBD_MASKS, allele_frequency, family_terms, pair_ibd, pair_indices


def _contig_evidence(store, contig, sample_ids, nthreads=None):
    """One contig's evidence arrays (genotype likelihoods float64 [N, L, 3], observed [N, L], positions [L]), decoded
    with ``nthreads`` CPUs (None: the store's allowance)."""
    file = store.contig(contig)
    recorded = tuple(map(str, file.get(evidence_records.EVIDENCE, items=("sample_ids",),
                                       nthreads=nthreads)["sample_ids"]))
    if recorded != tuple(sample_ids):
        raise ValueError(f"{contig}: the evidence record's sample axis differs from the run's")
    arrays = evidence_records.load_arrays(file, ("likelihoods", "observed", "positions"), nthreads=nthreads)
    return (np.asarray(arrays["likelihoods"], dtype=np.float64), np.asarray(arrays["observed"], dtype=np.bool_),
            np.asarray(arrays["positions"]))


def _contig_terms(store, contig, sample_ids, config: RelatednessConfig, genetic_map, pairs, posterior, loglik,
                  evidence=None, team=None):
    """One contig: the family model's site terms (returned) and the pairwise IBD chain (accumulated in place).
    ``evidence``: the contig's arrays (_contig_evidence) when already read; ``team()``: the Numba team for each
    parallel kernel, read just before it (None: the current team)."""
    gl, observed, positions = _contig_evidence(store, contig, sample_ids) if evidence is None else evidence
    if team is not None:
        execution.numba_threads_set(team())
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
    if team is not None:
        execution.numba_threads_set(team())
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


def _evidence_pass(store, contigs, sample_ids, config: RelatednessConfig, genetic_maps, pairs, posterior, loglik,
                   cores: int):
    """Every contig's family-model site terms, concatenated (C, logpi, logpost, support unit ranges, sites per
    chromosome), and the pairwise IBD chain accumulated in place (posterior, loglik), within ``cores`` CPUs.

    Release 3.4: with two or more CPUs, the next contig's evidence is read and decoded on one CPU (one helper thread,
    decoding with one thread) while the current contig is computed with a Numba team of cores - 1; a kernel that
    starts after that read has finished, and every kernel of the last contig, gets the whole team. The first contig
    (and every contig with one CPU) is read alone, with the store's allowance. At most two contigs' arrays are held."""
    child_terms, log_prior, log_posterior, ranges, chromosome_sites = [], [], [], [], []
    overlap = cores >= 2 and len(contigs) >= 2
    with ThreadPoolExecutor(max_workers=1) as reader:
        upcoming = None
        for index, contig in enumerate(contigs):
            evidence = upcoming.result() if upcoming is not None else _contig_evidence(store, contig, sample_ids)
            upcoming = (reader.submit(_contig_evidence, store, contigs[index + 1], sample_ids, 1)
                        if overlap and index + 1 < len(contigs) else None)

            def team(read=upcoming):
                return cores if read is None or read.done() else cores - 1
            terms, units = _contig_terms(store, contig, sample_ids, config, genetic_maps.for_contig(contig),
                                         pairs, posterior, loglik, evidence=evidence, team=team)
            del evidence
            lo = sum(sites.shape[1] for sites in child_terms)
            chromosome_sites.append(int(terms[0].shape[1]))
            ranges.extend((lo + int(a), lo + int(b)) for a, b in zip(units[:-1], units[1:]) if b > a)
            for collected, value in zip((child_terms, log_prior, log_posterior), terms):
                collected.append(value)
    C = np.ascontiguousarray(np.concatenate(child_terms, axis=1))
    logpi = np.ascontiguousarray(np.concatenate(log_prior, axis=0))
    logpost = np.ascontiguousarray(np.concatenate(log_posterior, axis=1))
    return C, logpi, logpost, ranges, chromosome_sites


def compute(store, contigs, sample_ids, config: RelatednessConfig, genetic_maps, budget) -> records.RelatednessProduct:
    """The product from the evidence records (reads only; ``run`` binds and stores it)."""
    contigs, sample_ids = tuple(contigs), tuple(sample_ids)
    started = time.monotonic()
    n = len(sample_ids)
    pairs = pair_indices(n, 0, n * (n - 1) // 2)
    # Column-major storage makes the final IBD means and site-count views contiguous for checkpoint writing.
    posterior = np.zeros((len(pairs), 4), order="F")
    loglik = np.zeros((len(pairs), len(IBD_MASKS)))
    with execution.numba_threads(budget.threads()):
        C, logpi, logpost, ranges, chromosome_sites = _evidence_pass(store, contigs, sample_ids, config, genetic_maps,
                                                                     pairs, posterior, loglik, budget.threads())
        execution.numba_threads_set(budget.threads())
        print(f"RELATEDNESS: {n} samples, {C.shape[1]} family sites over {len(contigs)} chromosomes "
              f"({len(ranges)} support units), {len(pairs)} pairs ({time.monotonic() - started:.0f} s)", flush=True)
        fit = family_model.fit(C, logpi, logpost, ranges, max_sweeps=config.max_sweeps,
                               log=lambda line: print(line, flush=True))
    # Preserve the pair/contig/site arithmetic, but reuse the posterior buffer for the released means.
    # A bounded denominator avoids another P-sized temporary; the product's views keep the buffer alive.
    for lo in range(0, len(pairs), 262144):
        block = posterior[lo:lo + 262144]
        block[:, :3] /= np.maximum(block[:, 3:4], 1.0)
    lr_ibd0, lr_ibd2 = loglik[:, 0] - loglik[:, 1], loglik[:, 0] - loglik[:, 2]
    del loglik
    family = np.full(n, -1, dtype=np.int64)
    for index, members in enumerate(fit.members):
        family[members] = index
    product = records.RelatednessProduct(
        sample_ids=sample_ids, contigs=contigs, pairs=pairs,
        ibd=posterior[:, :3], ibd_sites=posterior[:, 3],
        lr_ibd0=lr_ibd0, lr_ibd2=lr_ibd2, family=family,
        parents=fit.parents, member_support=fit.member_support, parent_support=fit.parent_support,
        half_sib=fit.half_sib, half_sib_gain=fit.half_sib_gain, loglik=fit.loglik, sweeps=fit.sweeps,
        converged=fit.converged, family_sites=int(C.shape[1]),
        chromosome_sites=tuple(chromosome_sites), support_units=len(ranges), search=fit.search,
        search_margin=fit.search_margin)
    sizes = np.bincount(family[family >= 0])
    print(f"RELATEDNESS: {len(fit.members)} families (largest {int(sizes.max())}), "
          f"{int(np.count_nonzero(fit.parents >= 0))} sequenced parent slots, {len(fit.half_sib)} half-sib links, "
          f"{int(np.count_nonzero(fit.member_support >= 0.75))} of {n} samples on their family on >= 75% of "
          f"support units; "
          f"{'converged' if fit.converged else 'sweep limit reached'} after {fit.sweeps} sweeps of the {fit.search} "
          f"search (margin {fit.search_margin:.1f}) ({time.monotonic() - started:.0f} s)", flush=True)
    return product
