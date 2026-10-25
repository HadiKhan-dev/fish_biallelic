"""Execution-only memory policy; none of these constants enters scientific identities."""
# At/above SOFT, admit one outer task after observing a full sampling interval since the previous start.
MEMORY_SOFT_FRACTION = 0.80
# At/above HARD, admit no outer work except one progress task if none is running or queued to start.
# Nested continuations always proceed so admitted work can finish and release memory.
MEMORY_HARD_FRACTION = 0.90
# One shared sample per run; also the minimum observation interval between outer starts near SOFT.
MEMORY_SAMPLE_SECONDS = 0.5


# Unchanged memory-ledger grant/release counters (and starts where recorded) with waiters for this many minutes,
# and less than MEMORY_STALL_CPU_CORES of process-tree CPU over the same window, fail the run.
MEMORY_STALL_MINUTES = 30.0
# The owner samples aligned ledger header counters without taking its lock at this interval.
MEMORY_STALL_POLL_SECONDS = 60.0
# CPU-seconds per elapsed second, summed across the run owner and its descendants.
MEMORY_STALL_CPU_CORES = 1.0
