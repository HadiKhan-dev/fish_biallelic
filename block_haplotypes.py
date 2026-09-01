"""Canonical Stage-1 block-haplotype discovery orchestrator.

Every informative block is reconstructed by the cap-free reversible-cavity
model. Allele-depth zero cells are missing observations, unsupported founder
alleles remain unknown, and there is no alternate Stage-1 inference backend.
This module owns only multiprocessing orchestration and re-exports the public
block result containers.
"""

import ctypes
import gc
import os

import numba
import numpy as np

import dynamic_threads
from multiprocessing_runtime import (
    ForkserverPool as _ForkserverPool,
    forkserver_context as _forkserver_ctx,
    main_module_guard as _main_module_guard,
)
from bhd_results import BlockResult, BlockResults
from bhd_reversible_cavity import ReversibleCavitySearchConfig

STAGE1_BACKEND = "reversible_cavity_depth_observation_v1"



def _nonnegative_env_int(name, default):
    """Read a non-negative operational tuning value from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value >= 0 else int(default)


# Retain reusable Numba/pattern-table arenas between blocks.  With the default
# 1.5-GiB per-worker threshold, even 112 workers remain well inside the normal
# 512-GiB production allocation; a 32-block periodic trim bounds fragmentation
# on smaller allocations and long-lived pools.  Both controls are operationally
# configurable without changing the scientific configuration or checkpoints.
_MALLOC_TRIM_RSS_BYTES = 1024 * 1024 * _nonnegative_env_int(
    "BHD_MALLOC_TRIM_RSS_MB", 1536
)
_MALLOC_TRIM_INTERVAL = _nonnegative_env_int(
    "BHD_MALLOC_TRIM_EVERY_BLOCKS", 32
)
_BLOCKS_SINCE_MALLOC_TRIM = 0
try:
    _RSS_PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    _RSS_PAGE_SIZE = 0


try:
    _libc = ctypes.CDLL("libc.so.6")

    def _trim_process_heap():
        _libc.malloc_trim(0)
except OSError:
    def _trim_process_heap():
        pass


def _current_process_rss_bytes():
    """Return current Linux RSS cheaply; zero when unavailable."""
    if not _RSS_PAGE_SIZE:
        return 0
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as handle:
            fields = handle.readline().split()
        return int(fields[1]) * _RSS_PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


def _maybe_malloc_trim(*, completed_block=False):
    """Trim only for high RSS or periodically after completed blocks.

    Avoid discarding reusable arenas after every block; trim only when the
    process is large or has completed the configured number of blocks.
    """
    global _BLOCKS_SINCE_MALLOC_TRIM
    if completed_block:
        _BLOCKS_SINCE_MALLOC_TRIM += 1
    over_rss_limit = (
        _MALLOC_TRIM_RSS_BYTES > 0
        and _current_process_rss_bytes() >= _MALLOC_TRIM_RSS_BYTES
    )
    periodic = (
        completed_block
        and _MALLOC_TRIM_INTERVAL > 0
        and _BLOCKS_SINCE_MALLOC_TRIM >= _MALLOC_TRIM_INTERVAL
    )
    if over_rss_limit or periodic:
        _trim_process_heap()
        _BLOCKS_SINCE_MALLOC_TRIM = 0




def _init_block_worker(
    total_cores,
    active_counter,
    extra_counter=None,
    started_counter=None,
    participant_counter=None,
    batch_generation=None,
    batch_task_count=None,
    startup_target=None,
    startup_ready=None,
):
    """Initializer for worker processes — sets up dynamic numba thread
    allocation based on number of currently-active workers.

    Wires dynamic_threads' shared dynamic-thread state, which is read by
    dynamic_threads.apply_dynamic_threads() at every
    phase boundary across the discovery kernels. That lets a straggler
    block grow into cores freed as its peers finish, instead of
    being pinned for its whole run to the thread count it got at start.
    extra_counter drives the remainder distribution (total threads in use ==
    total_cores, zero idle cores); None falls back to floor-only."""
    try:
        os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
        numba.config.NUMBA_NUM_THREADS = total_cores
        numba.set_num_threads(1)
    except Exception:
        pass
    # Every discovery phase boundary re-checks the same pool-wide state.
    dynamic_threads.set_dynamic_thread_state(
        total_cores,
        active_counter,
        extra_counter,
        started_counter,
        participant_counter,
        batch_generation,
        batch_task_count,
        startup_target,
        startup_ready,
    )

def _validate_block_parallelism(
    num_processes: int,
    total_numba_threads: int | None,
) -> tuple[int, int]:
    if (
        isinstance(num_processes, bool)
        or int(num_processes) != num_processes
        or int(num_processes) < 1
    ):
        raise ValueError("num_processes must be a positive integer")
    processes = int(num_processes)
    total_threads = (
        processes
        if total_numba_threads is None
        else total_numba_threads
    )
    if (
        isinstance(total_threads, bool)
        or int(total_threads) != total_threads
        or int(total_threads) < processes
    ):
        raise ValueError(
            "total_numba_threads must be an integer at least as large as "
            "num_processes"
        )
    total_threads = int(total_threads)
    available_cpus = len(os.sched_getaffinity(0))
    if processes > available_cpus or total_threads > available_cpus:
        raise ValueError(
            "block workers and Numba thread budget must lie within the "
            "current CPU affinity"
        )
    return processes, total_threads


class BlockDiscoveryPool:
    """Reusable block-worker pool with one bounded dynamic thread budget."""

    def __init__(
        self,
        num_processes: int,
        total_numba_threads: int | None = None,
    ) -> None:
        (
            self.num_processes,
            self.total_numba_threads,
        ) = _validate_block_parallelism(
            num_processes, total_numba_threads
        )
        self._closed = False
        self._active_counter = _forkserver_ctx.Value("i", 0)
        self._extra_counter = _forkserver_ctx.Value("i", 0)
        self._started_counter = _forkserver_ctx.Value("i", 0)
        self._participant_counter = _forkserver_ctx.Value("i", 0)
        self._batch_generation = _forkserver_ctx.Value("i", 0)
        self._batch_task_count = _forkserver_ctx.Value("i", 0)
        self._startup_target = _forkserver_ctx.Value("i", 1)
        self._startup_ready = _forkserver_ctx.Value("i", 0)
        self._batch_in_progress = False

        # Prevent forkserver workers from re-executing a pipeline entry point.
        with _main_module_guard():
            self._pool = _ForkserverPool(
                processes=self.num_processes,
                initializer=_init_block_worker,
                initargs=(
                    self.total_numba_threads,
                    self._active_counter,
                    self._extra_counter,
                    self._started_counter,
                    self._participant_counter,
                    self._batch_generation,
                    self._batch_task_count,
                    self._startup_target,
                    self._startup_ready,
                ),
            )

    @staticmethod
    def _store_counter(counter, value):
        with counter.get_lock():
            counter.get_obj().value = int(value)

    def _prepare_task_batch(self, n_tasks):
        if self._batch_in_progress:
            raise RuntimeError(
                "block-discovery pool already has an unfinished task batch"
            )
        if self._active_counter.value != 0:
            raise RuntimeError(
                "block-discovery workers from the preceding batch are still active"
            )
        task_count = int(n_tasks)
        initial_target = min(task_count, self.num_processes)
        self._store_counter(self._started_counter, 0)
        self._store_counter(self._participant_counter, 0)
        self._store_counter(self._batch_task_count, task_count)
        self._store_counter(self._startup_target, initial_target)
        self._store_counter(self._startup_ready, initial_target <= 1)
        with self._batch_generation.get_lock():
            generation = self._batch_generation.get_obj()
            generation.value += 1

    def imap_unordered(self, tasks):
        if self._closed:
            raise RuntimeError("block-discovery pool is closed")
        if not hasattr(tasks, "__len__"):
            tasks = tuple(tasks)
        self._prepare_task_batch(len(tasks))
        try:
            raw_results = self._pool.imap_unordered(
                _worker_generate_block_direct, tasks, chunksize=1
            )
        except Exception:
            raise
        self._batch_in_progress = True

        def consume_batch():
            completed = False
            try:
                for result in raw_results:
                    yield result
                completed = True
            finally:
                # An abandoned or failed iterator may still have queued work.
                # Keep the pool guarded in that case; close/terminate remains
                # available, but a new batch cannot corrupt the startup gate.
                if completed:
                    self._batch_in_progress = False

        return consume_batch()

    def close(self) -> None:
        if self._closed:
            return
        self._pool.close()
        self._pool.join()
        self._closed = True

    def terminate(self) -> None:
        if self._closed:
            return
        self._pool.terminate()
        self._pool.join()
        self._closed = True

    def __enter__(self) -> "BlockDiscoveryPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def _block_has_informative_retained_data(positions, reads, keep_flags):
    """Return whether a block can support founder discovery."""

    positions_value = np.asarray(positions)
    reads_value = np.asarray(reads)
    if reads_value.ndim != 3 or reads_value.shape[2] != 2:
        raise ValueError("reads must have shape (samples, sites, 2)")
    if positions_value.shape != (reads_value.shape[1],):
        raise ValueError("positions must match the reads site dimension")
    if reads_value.shape[0] < 1:
        raise ValueError("reads must contain at least one sample")
    if len(positions_value) == 0:
        return False
    retained = (
        np.ones(len(positions_value), dtype=np.bool_)
        if keep_flags is None
        else np.asarray(keep_flags) > 0
    )
    if retained.shape != (len(positions_value),):
        raise ValueError("keep_flags must match positions")
    if not np.any(retained):
        return False
    return bool(np.any(reads_value[:, retained, :] > 0))


def _worker_generate_block_direct(args):
    """Discover one informative block and return ``(input_index, result)``."""

    dynamic_threads.increment_active()
    dynamic_threads.apply_dynamic_threads()
    try:
        (
            block_index,
            positions,
            reads,
            keep_flags,
            discovery_config,
            discard_reads_after,
        ) = args
        if not _block_has_informative_retained_data(
            positions, reads, keep_flags
        ):
            return block_index, None

        from bhd_reversible_discovery import discover_block_reversible_cavity

        discovery = discover_block_reversible_cavity(
            positions,
            reads,
            keep_flags=keep_flags,
            config=discovery_config,
        )
        result = discovery.to_block_result(block_result_class=BlockResult)
        if discard_reads_after:
            result.reads_count_matrix = None
        _maybe_malloc_trim(completed_block=True)
        return block_index, result
    finally:
        dynamic_threads.release_dynamic_extra()
        dynamic_threads.decrement_active()


def generate_all_block_haplotypes(
    genomic_data,
    *,
    num_processes=16,
    discard_reads_after=True,
    total_numba_threads=None,
    block_pool=None,
    discovery_config=None,
):
    """Discover founder haplotypes in every informative input block.

    Empty blocks, blocks with no retained sites, and blocks with no reads at
    retained sites are omitted. ``discovery_config`` defaults to the canonical
    calibrated :class:`ReversibleCavitySearchConfig`.
    """

    from tqdm import tqdm

    config = (
        ReversibleCavitySearchConfig()
        if discovery_config is None
        else discovery_config
    )
    if not isinstance(config, ReversibleCavitySearchConfig):
        raise TypeError(
            "discovery_config must be a ReversibleCavitySearchConfig"
        )
    num_processes, total_numba_threads = _validate_block_parallelism(
        num_processes, total_numba_threads
    )
    if block_pool is not None:
        if not isinstance(block_pool, BlockDiscoveryPool):
            raise TypeError("block_pool must be a BlockDiscoveryPool")
        if (
            block_pool.num_processes != num_processes
            or block_pool.total_numba_threads != total_numba_threads
        ):
            raise ValueError(
                "block_pool worker and thread budgets must match this call"
            )

    tasks = []
    for index in range(len(genomic_data)):
        positions, reads, keep_flags = genomic_data[index]
        tasks.append((
            index,
            positions,
            reads,
            keep_flags,
            config,
            bool(discard_reads_after),
        ))

    def collect(pool):
        return list(tqdm(
            pool.imap_unordered(tasks),
            total=len(tasks),
            desc="Block haplotypes",
        ))

    if block_pool is None:
        with BlockDiscoveryPool(
            num_processes, total_numba_threads
        ) as local_pool:
            indexed_results = collect(local_pool)
    else:
        indexed_results = collect(block_pool)

    indexed_results.sort(key=lambda item: item[0])
    results = [
        result for _index, result in indexed_results if result is not None
    ]
    if discard_reads_after:
        gc.collect()
    return BlockResults(results)


def _selftest_stage1_orchestration():
    """Check canonical dispatch and deliberate degenerate-block omission."""

    import bhd_reversible_discovery as discovery_module

    class InlineBlockDiscoveryPool(BlockDiscoveryPool):
        def __init__(self):
            self.num_processes = 1
            self.total_numba_threads = 1
            self.tasks = []

        def imap_unordered(self, tasks):
            self.tasks.extend(tasks)
            for task in tasks:
                yield _worker_generate_block_direct(task)

    calls = []

    class FakeDiscovery:
        def __init__(self, positions, reads, keep_flags):
            self.positions = positions
            self.reads = reads
            self.keep_flags = keep_flags

        def to_block_result(self, *, block_result_class=None):
            result = block_result_class(
                self.positions,
                {0: np.ones((len(self.positions), 2), dtype=np.float64)},
                self.reads,
                keep_flags=self.keep_flags,
            )
            result.dispatch_marker = "reversible_cavity"
            return result

    def fake_discovery(
        positions,
        reads,
        keep_flags=None,
        *,
        config=None,
    ):
        calls.append((positions, reads, keep_flags, config))
        return FakeDiscovery(positions, reads, keep_flags)

    original = discovery_module.discover_block_reversible_cavity
    discovery_module.discover_block_reversible_cavity = fake_discovery
    try:
        positions = np.asarray([10, 20], dtype=np.int64)
        reads = np.zeros((1, 2, 2), dtype=np.int64)
        reads[0, :, 0] = 1
        flags = np.ones(2, dtype=np.int64)
        empty_positions = np.asarray([], dtype=np.int64)
        empty_reads = np.zeros((1, 0, 2), dtype=np.int64)
        no_kept = np.zeros(2, dtype=np.int64)
        all_missing = np.zeros_like(reads)
        genomic_data = (
            (positions, reads, flags),
            (empty_positions, empty_reads, np.asarray([], dtype=np.int64)),
            (positions, reads, no_kept),
            (positions, all_missing, flags),
        )
        config = ReversibleCavitySearchConfig()
        pool = InlineBlockDiscoveryPool()
        output = generate_all_block_haplotypes(
            genomic_data,
            num_processes=1,
            total_numba_threads=1,
            block_pool=pool,
            discard_reads_after=False,
            discovery_config=config,
        )
        assert len(pool.tasks) == 4
        assert len(calls) == 1
        assert calls[0][3] is config
        assert len(output) == 1
        assert output[0].dispatch_marker == "reversible_cavity"
        assert output[0].reads_count_matrix is reads
    finally:
        discovery_module.discover_block_reversible_cavity = original

    return {
        "canonical_dispatch": "pass",
        "degenerate_blocks_omitted": "pass",
        "exact_materialization_forwarding": "pass",
    }


__all__ = [
    "BlockDiscoveryPool",
    "BlockResult",
    "BlockResults",
    "ReversibleCavitySearchConfig",
    "STAGE1_BACKEND",
    "generate_all_block_haplotypes",
]
