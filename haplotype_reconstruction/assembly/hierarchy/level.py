"""One hierarchy level: partition, batch digests and reuse, joins (pool, in process, or batch queue),
and the level's output components.

Execution model (base hierarchy; memory admission through the run's MemoryGovernor): the pool's
workers attach the contig's float32 neutral evidence where the evidence view already lives in shared
memory (no copy per level). Each batch reserves its estimated peak (:func:`batch_memory_bytes`) before
it is sent to a worker and releases it when its result arrives, so batches wait for memory instead of
overcommitting it; the reservation is an envelope (the search's nested reservations, its concurrent
seed refits and a ladder extension, borrow its unused bytes first). The worker count is bounded by the
pending batches, the CPU budget and how many of the largest batches fit the memory limit together.
Workers share the budget's CPUs dynamically and are recycled after MAX_TASKS_PER_WORKER batches
(memory fragmentation). A single pending batch runs in the coordinating process with the whole budget
(holding its reservation). Every joined batch is recorded (``joins/<digest>``) by the coordinating
process as it completes; a batch whose digest any assembly of the contig already joined is reused,
never recomputed. Pacing changes timing only, never results.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import time

import numpy as np

from ...execution.batch_queue import BatchKind
from ...model.components import CountCall, CountStepDecision, FounderComponent
from ...store.identity import canonical_json, code_identity, jsonable
from .. import records as assembly_records
from .batch import (BatchTask, JoinResult, JoinSettings, batch_digest, join_batch, run_batch,
                    team_size)
from .batches import LevelPartition, partition, informative_masks

# Private worker residence retained between tasks, including loaded kernels, runtime and heap retention.
# Measured normal post-trim private maximum 1.994 GB includes retained task state; hold once per pool worker.
WORKER_RESIDENT_BYTES = 2048 << 20
MAX_TASKS_PER_WORKER = 4                 # base WORKER_MAXTASKS: recycle workers to return freed memory
BUNDLE_EVIDENCE_BYTES = 256 << 20        # batch queue: evidence transported per bundle (about)
BUNDLE_TASKS = 64                        # batch queue: tasks per bundle at most


@dataclass(frozen=True)
class LevelOutput:
    """components: the level's output in genomic order; plan: per batch (first, end, digest | None)
    over the level's input components (digest None: a pass-through batch); joins: per joined batch
    (batch order) its JoinSummary; reused: joined batches taken from existing join records;
    partition: the level's partition (phase breaks, informative masks, joint counts);
    elapsed_seconds: wall time of the level."""

    components: tuple[FounderComponent, ...]
    plan: tuple[tuple[int, int, str | None], ...]
    joins: tuple
    log_likelihoods: tuple[float, ...]
    reused: int
    partition: LevelPartition
    elapsed_seconds: float
    release_alternatives: tuple = ()
    shadows: tuple = ()


def join_context(settings: JoinSettings) -> str:
    """Canonical JSON of the settings and code every join of a level depends on."""
    return canonical_json(dict(
        code=code_identity("assembly"),
        linker=jsonable(settings.linker), join=jsonable(settings.join),
        count_rule=jsonable(settings.count_rule), cost_scale=settings.cost_scale,
        gap_horizon_scale=None if settings.level_one else settings.gap_horizon_scale,
        recombination_tolerance=None if settings.level_one else settings.recombination_tolerance,
        genetic_map=jsonable(settings.genetic_map.identity())))


def batch_memory_bytes(task: BatchTask) -> int:
    """Upper estimate of the peak bytes one batch's join allocates (base hierarchy._hmm_batch_memory_gb,
    over the batch's informative samples S instead of all samples).

    The batch keeps a float32 evidence slice of its samples over its sites; as much again covers
    transient copies (the search's flat emission layout while it is built). Linker scans keep seven
    float32 log emissions and seven float64 weights per sample and proxy site, plus uint8 pair codes per
    proxy site; its concurrent gap fits keep forward and backward pair messages (2 x 8 S K_b^2 per
    block and gap). The search keeps float64 binned emissions (S, K_b * (K_b + 1) // 2, m_b) with their static
    totals (counted 1.25 times: the batch's peak is close to it when it dominates), an int32 painting
    (S, sum m_b), painting back pointers bounded by tensor_budget_mb, its exclusion forwards
    (S, sum m_b) per excluded row (at most K + 1), one chunk of per-sample bounds, and move arrays and
    beam lists of O(B q K^2) entries up to the saturation ladder's normal top (quota 16 x 2^4); a
    search whose ladder climbs on to the backstop, and its concurrent seed refits, reserve their own
    bytes (join.search). The linker's concurrent steps reserve nothing beyond this estimate.
    Measured / estimate (tests/T4/k_scaling, estimate_ratios.py): K = 30, S = 320: L1 0.50-0.51, L2
    0.37-0.40, L3 0.55, L3 at 2,000 bins 0.80, a pool at quota 1024 (backstop counted) 0.69; AulStu chr3
    proxy L1 (K 8-16, S 290): 0.43-0.79. Worker private residence is held separately for the pool lifetime.
    """
    if task.branches:
        return max(batch_memory_bytes(replace(task, components=branch, branches=()))
                   for branch in task.branches)
    from ..join import emissions as join_emissions
    from ..join import moves as join_moves
    from ..join import search as join_search
    samples = len(task.samples)
    config = task.settings.join
    max_sites = task.settings.linker.max_sites
    site_counts = []
    founders = []
    for component in task.components:
        panels = task.prepared[component.start:component.stop]
        site_counts.append(sum(panel.sites for panel in panels))
        founders.append(component.founders)
    spb = join_emissions.sites_per_bin(site_counts, config.max_bins)
    span = sum(site_counts)
    total = config.tensor_budget_mb * 1024 ** 2 + 2 * 12 * samples * span
    total_bins = 0
    for sites, haps in zip(site_counts, founders):
        proxy_sites = min(sites, max_sites)
        ordered = haps * haps
        folded = haps * (haps + 1) // 2
        bins = math.ceil(sites / spb)
        total_bins += bins
        total += (samples * proxy_sites * 7 * 12 + proxy_sites * (ordered + folded)
                  + 10 * samples * folded * (bins + 1) + 4 * samples * bins)
    rows = max(founders)
    quota = config.paths_per_endpoint * 2 ** join_search.POOL_QUOTA_DOUBLINGS
    fits = (len(task.components) - 1) * 16 * samples * sum(k * k for k in founders)
    total += (8 * samples * total_bins * (rows + 2) + 8 * samples * join_moves.MOVES_PER_CHUNK
              + join_search.POOL_BYTES_PER_PATH_PAIR * len(task.components) * quota * rows * rows + fits)
    return int(total)


def _task(index, components, prepared, samples, settings) -> BatchTask:
    """A batch task with its components re-based on the prepared panels it covers."""
    first = components[0].start
    stop = components[-1].stop
    shifted = tuple(FounderComponent(c.start - first, c.stop - first, c.rows) for c in components)
    return BatchTask(index=index, components=shifted, prepared=tuple(prepared[first:stop]),
                     samples=samples, settings=settings)


def _encode_join(result: JoinResult) -> dict:
    return dict(schema=assembly_records.SCHEMA,
                rows=None if result.rows is None else np.ascontiguousarray(result.rows, np.int32),
                log_likelihood=result.log_likelihood,
                summary=None if result.summary is None else result.summary.record(),
                retained_inputs=result.retained_inputs,
                seconds=float(result.seconds), release_alternatives=result.release_alternatives,
                minimum_switches=result.minimum_switches, shadow=result.shadow)


def join_count_call(summary, alternatives=(), *, continuation_unverified=False) -> CountCall:
    """Count evidence of the latest actual join, without inventing a final count pass.

    Explicit death comparisons and accepted pool births retain their recorded margins.
    Seed/bound work remains in the join census. Evaluated different-K alternatives are
    count ambiguity; unperformed verification is recorded separately as an unknown call.
    """
    decisions = []
    for event in summary.events:
        stage = event.get("stage")
        if stage not in ("count_death", "count_birth") or "margin" not in event:
            continue
        death = stage == "count_death"
        decisions.append(CountStepDecision(
            kind="deletion" if death else "addition", founders=int(event["panel_size"]),
            accept_larger=not event["deleted"] if death else event.get("accepted") is not None,
            skipped_by_bound=False, margin=float(event["margin"]),
            band=tuple(event["band"]), ambiguous=bool(event["ambiguous"])))
    different_k = any(len({len(panel) for panel in family.get("panels", ())}) > 1
                      for family in alternatives if not family.get("count_unverified", False))
    ambiguous = (any(decision.ambiguous for decision in decisions) or different_k
                 or bool(summary.count.cycles) or bool(summary.count.final_paths_failing_death_test))
    unverified = (continuation_unverified or not summary.certified or summary.count.count_backstop_reached
                  or any(family.get("count_unverified", False) or family.get("unresolved_search", False)
                         for family in alternatives))
    return CountCall(founders=summary.final_panel_size, founders_before=summary.count.start_size,
                     ambiguous=True if ambiguous else None if unverified else False,
                     count_cycle=bool(summary.count.cycles),
                     cycle_undecided=bool(summary.count.cycles_undecided), decisions=tuple(decisions),
                     reason="join_search_unverified" if unverified else None)


def run_level(components, prepared, evidence, settings: JoinSettings, *, batch_size: int,
              min_informative: int, records=None, budget, queue=None,
              release_alternatives=()) -> LevelOutput:
    """Join one level (see the module docstring). ``records``: the assembly section's RecordScope
    already scoped to the level (``<P>/l<k>``) or None."""
    from ...execution import numba_threads
    from ..join.search import JoinSummary
    started = time.perf_counter()
    evidence = evidence.ensure_shared()
    from ..alternatives import component_choices, common_masks, common_site_masks, joint_choices
    choices, consumable, inherited = component_choices(components, release_alternatives)
    shared_sites, lost_sites = common_site_masks(choices, prepared) if consumable else ((), ())
    with numba_threads(team_size(budget)):
        masks = (common_masks(choices, prepared, evidence.neutral, settings.linker.max_sites,
                              usable_masks=shared_sites) if consumable else None)
        split = partition(components, prepared, evidence.neutral, batch_size=batch_size,
                          min_informative=min_informative, max_sites=settings.linker.max_sites, masks=masks)
    context = join_context(settings)
    tasks, digests = {}, {}
    retained = list(inherited)
    for index, (first, end) in enumerate(split.ranges):
        samples = split.batch_samples[index]
        source = [family for family in consumable
                  if components[first].start <= family["start"] and family["stop"] <= components[end - 1].stop]
        branches, combinations = joint_choices(choices[first:end])
        if end - first >= 2 and len(samples):
            task = _task(index, split.components[first:end], prepared, samples, settings)
            if source:
                task = replace(task, common_usable=shared_sites[first:end],
                               common_usable_lost=sum(lost_sites[first:end]))
            if source and combinations > 1:
                if branches:
                    origin = components[first].start
                    relative = tuple(tuple(FounderComponent(c.start - origin, c.stop - origin, c.rows)
                                           for c in branch) for branch in branches)
                    nominal_mask = np.logical_and.reduce(
                        informative_masks(components[first:end], prepared, evidence.neutral,
                                           settings.linker.max_sites))
                    task = replace(task, branches=relative, branch_sources=tuple(source),
                                   common_mask_lost=int(nominal_mask.sum()) - len(samples))
                else:
                    retained.extend(dict(family, unresolved_search=True, status="unresolved_joint_budget",
                                         joint_combinations=combinations,
                                         lost_usable_sites=task.common_usable_lost,
                                         usable_site_reference="union_of_input_alternatives") for family in source)
            tasks[index] = task
            digests[index] = batch_digest(task, context=context, evidence_identity=evidence.identity)
        else:
            retained.extend(source)
    plan = tuple((first, end, digests.get(index)) for index, (first, end) in enumerate(split.ranges))
    if records is not None and not records.has("plan"):
        records.save("plan", {"batches": [list(item) for item in plan]})
    stored = {} if records is None else assembly_records.load_joins(records.file, digests.values())
    results = {index: stored[digest] for index, digest in digests.items() if digest in stored}
    reused = len(results)
    pending = [tasks[index] for index in sorted(tasks) if index not in results]

    def record(result: JoinResult) -> None:
        payload = _encode_join(result)
        if records is not None:
            assembly_records.save_join(records.file, digests[result.index], payload)
        results[result.index] = payload

    estimates = [batch_memory_bytes(task) for task in pending]
    if queue is not None and len(pending) > 1:
        from ...execution.batch_queue import map_batches, queue_key
        key = queue_key(records, "join")
        if key is not None and reused:
            # Queue results are keyed by bundle number: another task subset is another stage.
            key = hashlib.sha256(json.dumps([key, sorted(results)]).encode()).hexdigest()
        for result in map_batches(HIERARCHY_BATCHES, pending, (evidence.scoring,), budget, queue=queue,
                                  key=key, worker_memory_gb=(max(estimates) + WORKER_RESIDENT_BYTES) / 1024 ** 3):
            record(result)
    elif len(pending) == 1:
        from ...execution import reserve_memory
        with reserve_memory(estimates[0], label="hierarchy", envelope=True):
            record(run_batch(pending[0], evidence.scoring, budget))
    elif pending:
        from ...execution import BudgetedPool, memory_worker_limit, reserve_memory
        workers = memory_worker_limit([size + WORKER_RESIDENT_BYTES for size in estimates],
                                      min(len(pending), int(budget.threads())), label="hierarchy")
        reference = evidence.shared.scoring
        with reserve_memory(workers * WORKER_RESIDENT_BYTES, label="hierarchy workers"):
            with BudgetedPool(budget, processes=workers, max_tasks_per_child=MAX_TASKS_PER_WORKER,
                              label="hierarchy") as pool:
                for result in pool.imap_unordered(join_batch, [(reference, task) for task in pending],
                                                  task_bytes=estimates, envelope=True):
                    record(result)
    output, joins, likelihoods, release_alternatives = [], [], [], retained
    shadows = []
    final_plan = list(plan)
    for index, (first, end) in enumerate(split.ranges):
        inputs = split.components[first:end]
        if index not in tasks:
            output.extend(inputs)      # a single component, or a batch without informative samples
            continue
        payload = results[index]
        if payload.get("shadow") is not None:
            shadows.append(dict(payload["shadow"], start=inputs[0].start, stop=inputs[-1].stop))
        release_alternatives.extend(dict(item, start=inputs[0].start + int(item["start"]),
                                          stop=inputs[0].start + int(item["stop"]))
                                    for item in payload.get("release_alternatives", ()))
        if payload.get('retained_inputs', False):
            release_alternatives.extend(family for family in consumable
                if inputs[0].start <= family['start'] and family['stop'] <= inputs[-1].stop)
            reason = 'unresolved_beam_materialization'
            output.extend(replace(component,
                                  break_before=component.break_before if at == 0 else reason,
                                  break_after=component.break_after if at + 1 == len(inputs) else reason)
                          for at, component in enumerate(inputs))
            # Reports use None to count all unchanged inputs. The attempted join
            # stays cached under its original digest with its diagnostic sidecar.
            final_plan[index] = (first, end, None)
            continue
        summary = JoinSummary.from_record(payload["summary"])
        output.append(FounderComponent(
            start=inputs[0].start, stop=inputs[-1].stop, rows=payload["rows"],
            break_before=inputs[0].break_before, break_after=inputs[-1].break_after,
            count_call=join_count_call(
                summary, payload.get("release_alternatives", ()),
                continuation_unverified=any(
                    family.get("unresolved_search", False)
                    and int(family["start"]) < inputs[-1].stop
                    and int(family["stop"]) > inputs[0].start for family in retained))))
        joins.append(summary)
        likelihoods.append(float(payload["log_likelihood"]))
    return LevelOutput(components=tuple(output), plan=tuple(final_plan), joins=tuple(joins),
                       log_likelihoods=tuple(likelihoods), reused=reused, partition=split,
                       elapsed_seconds=time.perf_counter() - started,
                       release_alternatives=tuple(release_alternatives), shadows=tuple(shadows))


def _task_sites(task: BatchTask) -> tuple[int, int]:
    """[first, end) contig site range the task reads."""
    return task.prepared[0].site_start, task.prepared[-1].site_start + task.prepared[-1].sites


def _bundles(tasks, arrays, worker_memory_gb):
    """Pack batch tasks with the evidence slice they read (about BUNDLE_EVIDENCE_BYTES, at most
    BUNDLE_TASKS tasks per bundle; a task is never split). Task site indices are re-based on the
    slice (the join reads evidence only through the panels' site_start)."""
    (scoring,) = arrays
    limit = max(1, BUNDLE_EVIDENCE_BYTES // (scoring.shape[0] * scoring.shape[2] * scoring.dtype.itemsize))
    group, left, right = [], None, None

    def package():
        shifted = [replace(task, prepared=tuple(replace(panel, site_start=panel.site_start - left)
                                                for panel in task.prepared)) for task in group]
        return dict(kind="hierarchy", tasks=shifted, arrays=(np.ascontiguousarray(scoring[:, left:right]),))

    for task in tasks:
        low, high = _task_sites(task)
        if group and (len(group) >= BUNDLE_TASKS or max(right, high) - min(left, low) > limit):
            yield package()
            group, left, right = [], None, None
        group.append(task)
        left = low if left is None else min(left, low)
        right = high if right is None else max(right, high)
    if group:
        yield package()


def _execute(payload, cpus):
    """Run one bundle of joins in the executing allocation (a BudgetedPool over its CPUs; the bundle's
    evidence slice is placed in shared memory once for its workers; batches reserve their estimates)."""
    from ...execution import BudgetedPool, memory_worker_limit, reserve_memory
    from ...execution.budget import as_budget
    from ...execution.shared_arrays import close_shared_memory, create_shared_array
    tasks = payload["tasks"]
    (scoring,) = payload["arrays"]
    estimates = [batch_memory_bytes(task) for task in tasks]
    planned = [size + WORKER_RESIDENT_BYTES for size in estimates]
    # This executor creates one shared scoring copy; include it once in every nonempty worker plan.
    planned[planned.index(max(planned))] += int(scoring.nbytes)
    workers = memory_worker_limit(planned, min(cpus, len(tasks)), label="hierarchy-queue")
    with reserve_memory(scoring.nbytes, label="hierarchy-queue evidence"):
        handle, reference = create_shared_array(scoring, copy_threads=cpus)
        try:
            with reserve_memory(workers * WORKER_RESIDENT_BYTES, label="hierarchy-queue workers"):
                with BudgetedPool(as_budget(cpus), processes=workers, max_tasks_per_child=MAX_TASKS_PER_WORKER,
                                  label="hierarchy-queue") as pool:
                    return list(pool.imap_unordered(join_batch, [(reference, task) for task in tasks],
                                                    task_bytes=estimates, envelope=True))
        finally:
            close_shared_memory([handle], unlink=True)


HIERARCHY_BATCHES = BatchKind("hierarchy", _bundles, _execute)
