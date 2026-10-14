"""Independent complete founder refinements in forkserver worker processes.

All passes for one component stay on the same worker process with a private
workspace (its own evidence columns, gathered from the chromosome's evidence in a
shared segment, models and messages). No component can change another's objective
or candidates. Results are restored to genomic order;
completion order affects only the available CPU budget: each worker's Numba thread
count is the node budget divided by the active workers (core.parallel's dynamic
threads, re-read at every sweep), so the tail components of a level take over the
cores of finished ones; a growing budget (a contig lane's lease) raises the shared
total while the level runs (core.parallel.live_total). Process x thread <= the budget.
The parent records every completed component (founder/checkpoints.py); workers
record long components' converged-sweep states.
"""
import gc
from functools import partial

from .candidates import completed_candidates
from .workspace import resolve_threads
from ...core import haplotypes, parallel


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
    """Pool initializer: one Numba thread until the shared dynamic budget is read. The
    refinement modules are imported here, while the parent is still sending the tasks."""
    import numba
    numba.set_num_threads(1)
    parallel.set_dynamic_thread_state(total_cores, active_counter, extra_counter,
                                      **startup_counters)
    from .. import founder_refinement  # noqa: F401
    from . import checkpoints, converged_sweep  # noqa: F401


def _dynamic_budget():
    """This worker's share of the node: floor(total / active workers) (+1 of the remainder)."""
    return parallel.get_dynamic_threads()


def _component_task(payload):
    """One component's converged refinement and count decisions, in a pool worker."""
    number, batch, component, shared, options = payload
    gc.collect()
    parallel.malloc_trim()
    # The chromosome's evidence and sites, attached from the parent's shared segments (no
    # copy); the component's workspace gathers its own columns from them.
    (evidence_handle, neutral), (sites_handle, sites) = map(parallel.attach_shared_array, shared)
    parallel.increment_active()
    try:
        return number, _refine_one(batch, component, neutral, sites, options, _dynamic_budget)
    finally:
        parallel.decrement_active()
        parallel.release_dynamic_extra()
        del neutral, sites
        parallel.close_shared_memory([evidence_handle, sites_handle])


def refine_independent_components(prepared, components, neutral, sites, *,
                                  config, num_threads, records, l1_blocks,
                                  cc_scale, count_additions=True):
    """records: the level's RefinementCheckpoints (founder/checkpoints.py), or None."""
    prepared = list(prepared)
    starts = {int(b.positions[0]): i for i, b in enumerate(prepared)}
    ends = {int(b.positions[-1]): i + 1 for i, b in enumerate(prepared)}
    completed = set() if records is None else records.completed()
    functions, indices = [], []
    answers = [None] * len(components)
    largest_sites = 0
    for number, block in enumerate(components):
        batch = prepared[starts[int(block.positions[0])]:ends[int(block.positions[-1])]]
        context = (None if l1_blocks is None else [b for b in l1_blocks
            if b.positions[0] >= block.positions[0] and b.positions[-1] <= block.positions[-1]])
        if number in completed:
            answers[number] = records.load(number)
            continue
        # Small components finish quickly: their completion record is enough
        # (thousands of per-proposal records would dominate L1 I/O). Long
        # components also record their resumable converged-sweep states.
        detail = (None if records is None or len(batch) <= config.window_blocks else
                  records.sweeps(number))
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
        # The worker numbered its single component 0; keep the chromosome index.
        answers[number] = (result[0][0], _renumber(result[1]["components"][0], number))
        if records is not None:
            records.save(number, *answers[number])

    if workers <= 1:
        with parallel.numba_thread_scope(total):
            for index, result in completed_candidates(functions, num_threads, per_component):
                record(indices[index], result)
    else:
        # The chromosome's evidence and sites reach the workers through shared segments (copied
        # once, in parallel) instead of a pickled copy of every component's columns sent one
        # task after another through the pool's task pipe.
        handles, shared = [], []
        with parallel.shared_memory_cleanup(handles):
            for array in (neutral, sites):
                handle, metadata = parallel.create_shared_array(array, copy_threads=total)
                handles.append(handle)
                shared.append(metadata)
            payloads = []
            for index, number in enumerate(indices):
                batch, block, _, _, options = functions[index].args
                payloads.append((number, batch, block, shared, options))
            _run_pool(payloads, workers, total, num_threads, record)
    from .converged_sweep import level_diagnostics
    output = haplotypes.BlockResults([block for block, _diagnostic in answers])
    return output, level_diagnostics(
        [diagnostic for _block, diagnostic in answers], count_additions)


def _run_pool(payloads, workers, total, num_threads, record):
    """The component tasks in a forkserver pool of ``workers`` processes (dynamic thread shares of
    the budget ``total``, live in a lane), largest component first; record(number, result)."""
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
    with parallel.live_total(num_threads) as live, parallel.main_module_guard():
        pool = parallel.NonDaemonicForkserverPool(
            processes=workers, initializer=_init_component_worker,
            initargs=(total, active, extra, dict(startup, live_total=live)))
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
