"""Pedigree stage entry: preparation, screening, the genome-wide panel, scoring, decision.

Chromosome workers keep each contig's prepared components, projections and screens resident across
the screen -> panel -> score barrier; every phase writes one per-contig record bound to its identity,
so an interrupted run resumes per contig. The decision, the full-marker exclusion release and the
report form the run record ``pedigree``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import execution
from ..evidence import records as evidence_records
from ..painting import records as painting_records
from ..store import CheckpointStore
from . import decision as decision_module
from . import exclusion, panel, prepare, records, scoring, tables
from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .records import NO_PARENT, GenomeScores, ParentEdges, PedigreeCalls, PedigreeProduct
from .resampling import minimum_contigs_for_tier

_GIB = 1 << 30


def tier_b_minimum_contigs(config: PedigreeConfig) -> int:
    """Smallest number of informative chromosomes for which Tier B can release a call.

    For the run precheck: with fewer, every resampled fit lacks the exposed-chromosome count of the
    exposure gate often enough that no configuration (not even M0) reaches the Tier-B support.
    """
    return minimum_contigs_for_tier(config.release.tier_b, config.exposure.minimum_exposed_contigs)


# --------------------------------------------------------------------------- chromosome handlers
def _open(common: dict, contig: str):
    store = CheckpointStore(common["root"], nthreads=execution.claim_threads())
    return store, store.contig(contig)


def _prepared_summary(file) -> tuple[int, str | None, int]:
    """(prepared components, omitted reason, painting components) without decoding arrays."""
    record = file.get(records.PREPARED,
                      items=("component_indices", "omitted_reason", "component_count"))
    return len(record["component_indices"]), record["omitted_reason"], int(record["component_count"])


def _prepare_handler(contig, state, phase, common, budget):
    """Prepare and record one contig (no work when its record is current)."""
    store, file = _open(common, contig)
    config, chromosome_map = common["config"], common["genetic_maps"].for_contig(contig)
    store.bind_identity(file, records.PREPARED,
                        records.prepared_identity(file, config, chromosome_map))
    if not file.has(records.PREPARED):
        painting = painting_records.load(file, panels=False, emissions=True)
        evidence = evidence_records.load(file, nthreads=execution.claim_threads())
        prepared = prepare.prepare_chromosome(painting, evidence, chromosome_map,
                                              config.preparation, execution.claim_threads)
        del painting, evidence
        file.put(records.PREPARED, records.encode_prepared(prepared))
    return None, None


def _array_bytes(value, seen=None) -> int:
    """Bytes of the NumPy arrays reachable from ``value`` (dataclass fields, attributes, sequences, mappings),
    each object counted once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return 0
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list, set, frozenset)):
        children = value
    elif hasattr(value, "__dict__"):
        children = vars(value).values()
    elif hasattr(value, "__dataclass_fields__"):
        children = [getattr(value, name) for name in value.__dataclass_fields__]
    else:
        return 0
    return sum(_array_bytes(child, seen) for child in children)


def _score_handler(contig, state, phase, common, budget):
    """'screen': M0/M1 per component (screen record); 'score': M2, exposure, junctions.

    The prepared record and the resident scoring arrays are kept across the screen -> score barrier only when
    their exact bytes can be reserved at once (execution.memory, no wait); otherwise the score phase rebuilds them
    from the prepared record, as a resumed run does (identical results). The score call reserves its own bytes
    (the executor's score phase holds none): a retained state's reservation lends them first (its arrays are
    reused), so only the rest is taken from the run's memory."""
    store, file = _open(common, contig)
    config, eligibility = common["config"], common["eligibility"]
    if phase == "screen":
        store.bind_identity(file, records.SCREEN, records.screen_identity(file, config, eligibility))
        prepared = resident = hold = None
        if file.has(records.SCREEN):
            screen = records.decode_screen(file.get(records.SCREEN))
        else:
            prepared = records.decode_prepared(file.get(records.PREPARED))
            screen, resident = scoring.screen_chromosome(prepared, eligibility, config,
                                                         execution.claim_threads)
            file.put(records.SCREEN, records.encode_screen(screen))
            governor = execution.memory_governor()
            hold = governor.request(_array_bytes((prepared, resident)), label="PEDIGREE SCORING retained",
                                    envelope=True)
            if hold.poll():
                governor.rest(hold.id)          # resident memory, not running work
            else:
                hold.release()
                hold = prepared = resident = None
        return (prepared, screen, resident, hold), (screen.pair_scores(), screen.informative_markers)
    prepared, screen, resident, hold = state
    work = None
    try:
        candidate_panel = common["panel"]
        store.bind_identity(file, records.SCORES, records.scores_identity(file, candidate_panel))
        if file.has(records.SCORES):
            return None, records.decode_scores(file.get(records.SCORES))
        work = execution.memory_governor().request(_scoring_bytes(store, contig), label="PEDIGREE SCORING score call",
                                                   parent=None if hold is None else hold.id)
        work.wait()
        if prepared is None:
            prepared = records.decode_prepared(file.get(records.PREPARED))
        scores = scoring.score_chromosome(prepared, screen, resident, candidate_panel.trios,
                                          eligibility, config, execution.claim_threads)
        file.put(records.SCORES, records.encode_scores(scores))
        return None, scores
    finally:
        if work is not None:
            work.release()
        if hold is not None:
            hold.release()


def _exclusion_handler(contig, state, phase, common, budget):
    """Full-marker exclusion evidence of one contig for the target dyads."""
    store, file = _open(common, contig)
    config, dyads = common["config"], common["dyads"]
    store.bind_identity(file, records.EXCLUSION, records.exclusion_identity(file, config, dyads))
    if file.has(records.EXCLUSION):
        return None, records.decode_exclusion(file.get(records.EXCLUSION))
    evidence = evidence_records.load(file, nthreads=execution.claim_threads())
    value = exclusion.score_contig(contig, evidence.likelihoods, evidence.observed,
                                   evidence.positions, dyads, config.exclusion,
                                   execution.claim_threads)
    del evidence
    file.put(records.EXCLUSION, records.encode_exclusion(value))
    return None, value


# --------------------------------------------------------------------------- memory planning
def _preparation_bytes(store, contig: str, n_samples: int) -> int:
    """Raw evidence, painting, three N B S^2 float64 arrays (emissions, factors, workspace), 2 GiB."""
    file = store.contig(contig)
    summary = painting_records.load_summary(file)
    states = sum(8 * n_samples * shape.bins * shape.states ** 2 for shape in summary.components)
    return (file.info(evidence_records.EVIDENCE).decoded_bytes * 5 // 4
            + file.info(painting_records.KEY).decoded_bytes + 3 * states + 2 * _GIB)


def _scoring_bytes(store, contig: str) -> int:
    """Four prepared records (factors, marginals, projection, kernel workspaces) and 2 GiB."""
    return 4 * store.contig(contig).info(records.PREPARED).decoded_bytes + 2 * _GIB


def _exclusion_bytes(store, contig: str) -> int:
    """The decoded evidence, the int16 pattern codes per sample and site, and 2 GiB."""
    return store.contig(contig).info(evidence_records.EVIDENCE).decoded_bytes * 5 // 4 + 2 * _GIB


# --------------------------------------------------------------------------- product views
def _calls(tier, rows: np.ndarray) -> PedigreeCalls:
    parents = np.full((len(tier.state), 2), NO_PARENT, dtype=np.int64)
    for child, row in enumerate(tier.row):
        if row >= 0:
            named = [int(parent) for parent in rows[row, 1:] if parent >= 0]
            parents[child, :len(named)] = named
    return PedigreeCalls(tier.state.copy(), parents, tier.status)


def _edges(partial, sample_ids) -> ParentEdges:
    """Released parent edges in (child, slot) order (base refinement.model.pedigree_edges)."""
    children, slots = np.nonzero(partial.supported)
    return ParentEdges(tuple(sample_ids), partial.parents[children, slots].astype(np.int64),
                       children.astype(np.int64), slots.astype(np.int64),
                       partial.count_minimum.copy(), partial.count_maximum.copy())


# --------------------------------------------------------------------------- the stage
def run(store, contigs: tuple[str, ...], sample_ids: tuple[str, ...],
        eligibility: ParentEligibility | None, config: PedigreeConfig, genetic_maps,
        budget) -> PedigreeProduct:
    """Infer the genome-wide pedigree; returns (and stores) the run record ``pedigree``.

    store:        the run's CheckpointStore. Reads per contig the painting record and the evidence
                  record (through their stages' records API); writes pedigree/prepared, /screen,
                  /scores and /exclusion per contig and ``pedigree`` in the run file.
    contigs:      configured contigs in order (all must be painted).
    sample_ids:   the run's sample axis (N); every per-sample array uses this order.
    eligibility:  caller design assumptions over ``sample_ids``; None = unrestricted (simulations).
    config:       PedigreeConfig.
    genetic_maps: GeneticMapSet; the meiotic switch probability of the transmission selectors.
    budget:       the CPU budget (execution.CpuBudget) for this stage.
    """
    contigs, sample_ids = tuple(contigs), tuple(sample_ids)
    if len(sample_ids) < 3:
        raise ValueError("pedigree inference needs at least three samples")
    eligibility = eligibility or ParentEligibility.unrestricted(sample_ids)
    if eligibility.sample_ids != sample_ids:
        raise ValueError("parent eligibility sample order differs from the run's samples")
    store.require(painting_records.KEY, contigs)
    for contig in contigs:
        painting_records.require_evidence_binding(store.contig(contig))
    n = len(sample_ids)
    common = dict(root=store.root, config=config, eligibility=eligibility,
                  genetic_maps=genetic_maps)

    # Preparation (chromosome workers; a current record is reused). Each task reserves its estimate.
    estimates = [_preparation_bytes(store, c, n) for c in contigs]
    limit = execution.memory_worker_limit(estimates, len(contigs), label="PEDIGREE PREPARATION")
    weights = [store.contig(c).info(evidence_records.EVIDENCE).stored_bytes for c in contigs]
    with execution.ChromosomeExecutor(contigs, _prepare_handler, budget=budget, weights=weights,
                                      max_workers=limit, label="PEDIGREE PREPARATION",
                                      task_bytes=estimates) as pool:
        pool.run("prepare", common)
    summaries = {contig: _prepared_summary(store.contig(contig)) for contig in contigs}
    informative = tuple(contig for contig in contigs if summaries[contig][0])
    omitted = {contig: summaries[contig][1] for contig in contigs if not summaries[contig][0]}
    if len(informative) < config.exposure.minimum_exposed_contigs:
        raise ValueError(
            f"pedigree inference needs evidence on at least {config.exposure.minimum_exposed_contigs} "
            f"chromosomes (found {len(informative)}); Tier B needs at least "
            f"{tier_b_minimum_contigs(config)}")

    # Screen, genome-wide panel, score (projections stay resident across the barrier when their memory can be
    # reserved; _score_handler).
    estimates = [_scoring_bytes(store, c) for c in informative]
    limit = execution.memory_worker_limit(estimates, len(informative), label="PEDIGREE SCORING")
    weights = [store.contig(c).info(records.PREPARED).stored_bytes for c in informative]
    with execution.ChromosomeExecutor(informative, _score_handler, budget=budget, weights=weights,
                                      max_workers=limit, label="PEDIGREE SCORING",
                                      task_bytes=estimates) as pool:
        screens = pool.run("screen", common)
        markers = np.asarray([value[1] for value in screens], dtype=np.float64)
        screen_scores = panel.robust_screen(np.stack([value[0] for value in screens]), markers,
                                            eligibility, config)
        candidate_panel = panel.fixed_panel(screen_scores, eligibility, config)
        chromosome_scores = tuple(pool.run("score", dict(common, panel=candidate_panel), task_bytes=0))

    identity = records.product_identity(store, contigs, informative, config, eligibility,
                                        candidate_panel)
    store.bind_identity(store.run, records.PRODUCT, identity)
    if store.run.has(records.PRODUCT):
        return records.load_product(store)

    # Decision, then the full-marker count release of unresolved Tier-B children.
    scores = GenomeScores.stack(sample_ids, candidate_panel, chromosome_scores)
    decided = decision_module.decide(scores, eligibility, config, budget)
    rows = decided.alternatives.rows
    dyads = exclusion.target_dyads(decided.tier_b, rows, eligibility)
    exclusion_summary = dict(
        model="full-marker-mendelian-directed-v1", alpha=config.exclusion.alpha,
        genotype_replacement_probability=config.exclusion.genotype_replacement_probability,
        conditional_on_supported_edge_directions=True, candidate_dyads=int(len(dyads)),
        contigs=len(contigs), status="no_unresolved_candidates")
    log_e, observed = np.empty(0), np.empty(0, dtype=np.int64)
    if len(dyads):
        weights = [store.contig(c).info(evidence_records.EVIDENCE).stored_bytes for c in contigs]
        estimates = [_exclusion_bytes(store, c) for c in contigs]
        limit = execution.memory_worker_limit(estimates, len(contigs), label="PEDIGREE EXCLUSION")
        with execution.ChromosomeExecutor(contigs, _exclusion_handler, budget=budget, weights=weights,
                                          max_workers=limit, label="PEDIGREE EXCLUSION",
                                          task_bytes=estimates) as pool:
            evidence = pool.run("exclusion", dict(common, dyads=dyads))
        log_e, observed = exclusion.combine(list(evidence))
        exclusion_summary.update(status="evaluated", chromosomes=[dict(
            contig=value.contig, markers=value.markers,
            pattern_fallback_samples=value.pattern_fallback_samples) for value in evidence])
    released = exclusion.release_counts(
        decided.tier_b, decided.partial_b, decided.complete, decided.evidence.graph_conflict,
        decided.informative, rows, sample_ids, eligibility, dyads, log_e, observed, config)
    exclusion_summary["released"] = int(np.count_nonzero(released.released))

    calls = _calls(released.tier_b, rows)
    search = panel.search_diagnostics(candidate_panel, eligibility, calls.parents, sample_ids)
    by_contig = {value.contig: value for value in chromosome_scores}
    chromosome_summary = pd.DataFrame([dict(
        Contig=contig, ComponentCount=summaries[contig][2],
        ScoredComponentCount=summaries[contig][0],
        InformativeMarkers=by_contig[contig].informative_markers if contig in by_contig else 0,
        OmittedReason=omitted.get(contig)) for contig in contigs])
    product = PedigreeProduct(
        sample_ids=sample_ids, contigs=contigs, informative_contigs=informative, omitted=omitted,
        calls=calls, edges=_edges(released.partial_b, sample_ids),
        tier_a=_calls(decided.tier_a, rows), complete=_calls(decided.complete, rows),
        reach=decided.reach, calibration=decided.calibration, family_solve=decided.family_solve,
        exclusion=exclusion_summary, eligibility=eligibility.summary(), panel=candidate_panel,
        report=tables.build(decided, released, search, chromosome_summary, config),
        tie_events=decided.tie_events)
    store.run.put(records.PRODUCT, records.encode_product(product))
    store.mark_stage_complete(records.STAGE)
    print(f"PEDIGREE: {decided.reach.statement}", flush=True)
    return product
