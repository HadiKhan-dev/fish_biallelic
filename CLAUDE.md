@AGENTS.md

Claude and its subagents must follow the repository's full-allocation CPU
policy. For verified CPU-bound work, use the complete verified Slurm CPU
affinity without reserving dedicated CPUs for orchestration, the agent, the
operating system, or filesystem activity. Keep the aggregate process-count
times threads-per-process within the allocation, and reduce concurrency only
for demonstrated task-count, memory-capacity, memory-bandwidth, or I/O limits.

A focused post-implementation audit and targeted correctness checks gate every
expensive launch. After that launch gate passes, deeper validation gates
acceptance rather than speculative execution.

During authorized long-running work, reassess CPU utilization at natural
execution boundaries and expand safe CPU-bound work into idle allocated cores.
