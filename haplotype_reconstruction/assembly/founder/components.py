"""Independent complete founder refinements in forkserver worker processes.

All passes for one component stay on the same worker process with a private
workspace (its own evidence slice, models and messages). No component can change
another's objective or candidates. Results are restored to genomic order;
completion order affects only the available CPU budget: each worker's Numba thread
count is the node budget divided by the active workers (core.parallel's dynamic
threads, re-read at every sweep), so the tail components of a level take over the
cores of finished ones. Process x thread <= the budget.
"""
import copy
import gc
from functools import partial

import numpy as np

from ..checkpoints import AssemblyCheckpointStore
from .candidates import completed_candidates
from .workspace import resolve_threads
from ...core import haplotypes, parallel


class ScopedCheckpoints:
    def __init__(self, store, prefix):
        self.store, self.prefix = store, prefix

    def load(self, phase):
        return self.store.load(f"{self.prefix}.{phase}")

    def save(self, phase, payload):
        self.store.save(f"{self.prefix}.{phase}", payload)


def scoped_checkpoints(store, prefix):
    """Give concurrent components distinct files and single-threaded small I/O."""
    if store is None:
        return None
    if isinstance(store, AssemblyCheckpointStore):
        store = copy.copy(store)
        store.checkpoint_store = copy.copy(store.checkpoint_store)
        store.checkpoint_store.nthreads = 1
    return ScopedCheckpoints(store, prefix)


def _renumber(value, number):
    # Component-local passes number their singleton 0. Preserve the chromosome
    # index in every diagnostic without changing founder/deletion indices.
    if isinstance(value, dict):
        return {key: number if key == "component" else _renumber(item, number)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_renumber(item, number) for item in value]
    return value


def _refine_one(batch, component, neutral, sites, options, budget):
    from ..founder_refinement import _refine_serial_components
    return _refine_serial_components(batch, haplotypes.BlockResults([component]),
        neutral, sites, **dict(options, num_threads=budget or options["num_threads"]))


def _init_component_worker(total_cores, active_counter, extra_counter, startup_counters):
    """Pool initializer: one Numba thread until the shared dynamic budget is read."""
    import numba
    numba.set_num_threads(1)
    parallel.set_dynamic_thread_state(total_cores, active_counter, extra_counter,
                                      **startup_counters)


def _dynamic_budget():
    """This worker's share of the node: floor(total / active workers) (+1 of the remainder)."""
    return parallel.get_dynamic_threads()


def _component_task(payload):
    """One component's converged refinement and count decisions, in a pool worker."""
    number, batch, component, neutral, sites, options = payload
    gc.collect()
    parallel.malloc_trim()
    parallel.increment_active()
    try:
        return number, _refine_one(batch, component, neutral, sites, options, _dynamic_budget)
    finally:
        parallel.decrement_active()
        parallel.release_dynamic_extra()


def _evidence_slice(neutral, sites, component):
    """The component's columns of the chromosome evidence (the workspace gathers the same
    values from them)."""
    lo = int(np.searchsorted(sites, component.positions[0]))
    hi = int(np.searchsorted(sites, component.positions[-1])) + 1
    return np.ascontiguousarray(neutral[:, lo:hi]), np.ascontiguousarray(sites[lo:hi])


def refine_independent_components(prepared, components, neutral, sites, *,
                                  config, num_threads, checkpoints, l1_blocks,
                                  cc_scale, count_additions=True):
    prepared = list(prepared)
    starts = {int(b.positions[0]): i for i, b in enumerate(prepared)}
    ends = {int(b.positions[-1]): i + 1 for i, b in enumerate(prepared)}
    from .checkpoints import FounderCheckpointStore
    completed_store = (None if checkpoints is None else FounderCheckpointStore(
        scoped_checkpoints(checkpoints, "completed"), prepared))
    functions, indices = [], []
    answers = [None] * len(components)
    largest_sites = 0
    for number, block in enumerate(components):
        batch = prepared[starts[int(block.positions[0])]:ends[int(block.positions[-1])]]
        context = (None if l1_blocks is None else [b for b in l1_blocks
            if b.positions[0] >= block.positions[0] and b.positions[-1] <= block.positions[-1]])
        cached = None if completed_store is None else completed_store.load(f"component{number}")
        if cached is not None:
            answers[number] = (haplotypes.BlockResults([cached["block"]]), cached["diagnostic"])
            continue
        # Small components finish quickly. A compact completion checkpoint is
        # enough; thousands of per-proposal files would dominate L1 I/O. Long
        # components retain their ordinary resumable passes and iterations.
        detail = (None if len(batch) <= config.window_blocks else
                  scoped_checkpoints(checkpoints, f"component{number}"))
        options = dict(config=config, num_threads=num_threads, cc_scale=cc_scale,
            checkpoints=detail,
            l1_blocks=context, count_additions=count_additions)
        indices.append(number)
        functions.append(partial(_refine_one, batch, block, neutral, sites, options))
        largest_sites = max(largest_sites, len(block.positions))
    # Private evidence slices, models, logs, messages and traceback scale with the
    # component; include the worker interpreter and ample native workspace headroom.
    per_component = 1024 ** 3 + 160 * len(neutral) * largest_sites
    total = int(resolve_threads(num_threads))
    workers = min(len(functions), total)
    from ...painting.model import available_process_memory_bytes
    available = available_process_memory_bytes()
    if available is not None:
        workers = min(workers, max(1, int(available // per_component)))

    def record(number, result):
        answers[number] = result
        if completed_store is not None:
            completed_store.save(f"component{number}", {
                "block": result[0][0], "diagnostic": result[1]})

    if workers <= 1:
        with parallel.numba_thread_scope(total):
            for index, result in completed_candidates(functions, num_threads, per_component):
                record(indices[index], result)
    else:
        payloads = []
        for index, number in enumerate(indices):
            batch, block, _, _, options = functions[index].args
            payloads.append((number, batch, block, *_evidence_slice(neutral, sites, block), options))
        context = parallel.forkserver_context
        active, extra = context.Value('i', 0), context.Value('i', 0)
        # Distinct-worker startup gate (as the hierarchy pools): no early worker takes
        # the whole node while its peers are still starting.
        startup = dict(started_counter=context.Value('i', 0),
                       participant_counter=context.Value('i', 0),
                       batch_generation=context.Value('i', 0),
                       batch_task_count=context.Value('i', len(payloads)),
                       startup_target=context.Value('i', workers),
                       startup_ready=context.Value('i', 0))
        # Largest components first: the level's tail is the smallest work.
        order = sorted(range(len(payloads)), key=lambda i: -len(payloads[i][2].positions))
        with parallel.main_module_guard():
            pool = parallel.NonDaemonicForkserverPool(
                processes=workers, initializer=_init_component_worker,
                initargs=(total, active, extra, startup))
            try:
                for number, result in pool.imap_unordered(
                        _component_task, [payloads[i] for i in order]):
                    record(number, result)
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
    output = haplotypes.BlockResults([result[0][0] for result in answers])
    diagnostics = dict(answers[0][1], components=[
        _renumber(result[1]["components"][0], number)
        for number, result in enumerate(answers)])
    return output, diagnostics
