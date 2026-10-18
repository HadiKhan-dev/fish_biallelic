"""How the package uses CPUs, memory and processes; no scientific code.

Public names (imported on first access, so the package bootstrap stays light):

budget          CpuBudget, claim_threads, numba_threads
pools           BudgetedPool
chromosome_workers  ChromosomeExecutor, memory_worker_limit
shared_arrays   SharedArrayRef, create_shared_array, empty_shared_array, new_segment, attach_shared_array,
                close_shared_memory
resources       affinity_cpus, available_memory_bytes, resident_bytes
memory          memory_governor, reserve_memory, memory_statistics, MemoryGovernor, malloc_trim, release_memory
run_record      timed_stage, stage_timer, record_product, worker_stages
bootstrap       profiling, start_run

Driver-level modules are imported explicitly: execution.contig_lanes, execution.kernel_cache,
execution.batch_queue, execution.run_log.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "CpuBudget": "budget", "claim_threads": "budget", "numba_threads": "budget",
    "BudgetedPool": "pools",
    "ChromosomeExecutor": "chromosome_workers", "memory_worker_limit": "chromosome_workers",
    "SharedArrayRef": "shared_arrays", "create_shared_array": "shared_arrays",
    "empty_shared_array": "shared_arrays", "new_segment": "shared_arrays",
    "attach_shared_array": "shared_arrays", "close_shared_memory": "shared_arrays",
    "affinity_cpus": "resources", "available_memory_bytes": "resources", "resident_bytes": "resources",
    "malloc_trim": "memory", "release_memory": "memory", "memory_governor": "memory",
    "reserve_memory": "memory", "memory_statistics": "memory", "MemoryGovernor": "memory",
    "timed_stage": "run_record", "stage_timer": "run_record", "record_product": "run_record",
    "worker_stages": "run_record",
    "profiling": "bootstrap", "start_run": "bootstrap",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
