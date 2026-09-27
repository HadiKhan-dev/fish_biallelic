"""Transport adapters for the ordinary local-fit and hierarchy callbacks.

The queue owns scheduling and result publication. This module only packs the
arrays a bundle needs and invokes the same workers as node-local execution.
Task IDs, marker coordinates, observation masks and model parameters survive
transport; CPU budgets are taken from the executing allocation.
"""
import numpy as np

from . import parallel, runtime


def bundles(kind, tasks, arrays, worker_memory_gb):
    """Bound transported evidence to about 256 MiB; never split a science task.

    Local fitting uses an observation mask as its second array and indexes
    markers relative to the transported slice. Hierarchy tasks instead carry
    genomic blocks and use a sliced physical-position array.
    """
    evidence, other = arrays
    bytes_per_site = evidence.shape[0] * evidence.shape[2] * evidence.dtype.itemsize
    if kind == 'local_fit':
        bytes_per_site += other.shape[0] * other.dtype.itemsize
    limit = max(1, (256 << 20) // bytes_per_site)
    cap = 256 if kind == 'local_fit' else 64
    group, left, right = [], None, None

    def bounds(task):
        if kind == 'local_fit':
            return int(task[2][0]), int(task[2][-1]) + 1
        blocks = task[3]
        return (
            int(np.searchsorted(other, blocks[0].positions[0])),
            int(np.searchsorted(other, blocks[-1].positions[-1])) + 1,
        )

    def package():
        if kind == 'local_fit':
            adjusted = [(t[0], t[1], t[2] - left, *t[3:]) for t in group]
            second = np.ascontiguousarray(other[:, left:right])
        else:
            adjusted = list(group)
            second = np.ascontiguousarray(other[left:right])
        return dict(
            kind=kind,
            tasks=adjusted,
            arrays=(np.ascontiguousarray(evidence[:, left:right]), second),
            worker_memory_gb=worker_memory_gb,
        )

    for task in tasks:
        lo, hi = bounds(task)
        if group and (len(group) >= cap or max(right, hi) - min(left, lo) > limit):
            yield package()
            group, left, right = [], None, None
        group.append(task)
        left = lo if left is None else min(left, lo)
        right = hi if right is None else max(right, hi)
    if group:
        yield package()


def execute(payload, cores):
    """Run a bundle with allocation-local shared arrays and dynamic CPU sharing.

    Numerical callbacks and their initializers remain in their scientific
    modules. Import them here, after the helper CLI has set the CPU ceiling.
    Shared arrays are released on both success and worker failure.
    """
    tasks = payload['tasks']
    handles, metadata = [], []
    try:
        for array in payload['arrays']:
            handle, meta = parallel.create_shared_array(array, copy_threads=cores)
            handles.append(handle)
            metadata.append(meta)
        memory = max(1., float(payload['worker_memory_gb'])) * (1024**3)
        workers = max(
            1, min(cores, len(tasks), int(runtime.available_memory_bytes() / memory))
        )
        context = parallel.forkserver_context
        active, extra = context.Value('i', 0), context.Value('i', 0)
        startup = {
            name: context.Value('i', value)
            for name, value in dict(
                started_counter=0,
                participant_counter=0,
                batch_generation=0,
                batch_task_count=len(tasks),
                startup_target=workers,
                startup_ready=0,
            ).items()
        }
        if payload['kind'] == 'local_fit':
            from ..discovery import feedback, path_blocks

            initialize, function = feedback.initialize, path_blocks.worker
            shared = metadata
        elif payload['kind'] == 'hierarchy':
            from ..assembly import hierarchy

            initialize, function = hierarchy._init_worker_meta, hierarchy._process_single_batch
            shared = dict(probs=metadata[0], sites=metadata[1])
            # Tuple field 15 is the existing hierarchy callback's thread budget.
            # Never inherit the submitting node's process-by-thread allocation.
            tasks = [
                (*task[:15], max(1, cores // workers), *task[16:])
                for task in tasks
            ]
        else:
            raise ValueError(f'Unknown distributed task kind: {payload["kind"]}')
        with parallel.main_module_guard():
            with parallel.NonDaemonicForkserverPool(
                workers,
                initializer=initialize,
                initargs=(shared, cores, active, extra, startup),
            ) as pool:
                return list(pool.imap_unordered(function, tasks, chunksize=1))
    finally:
        parallel.close_shared_memory(handles, unlink=True)
