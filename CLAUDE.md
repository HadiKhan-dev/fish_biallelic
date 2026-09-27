# Claude Code repository instructions

@AGENTS.md

`AGENTS.md` is the canonical repository guidance, including the user's standing
authorization for autonomous CSD3 allocation management. Apply its account
preferences, aggregate 448-CPU cap across SL3-CPU and SL2-CPU combined,
SL3-GPU-only policy, allocation reuse, and durable two-hour idle cleanup to
Claude Code as well. SL4-CPU allocations are excluded from the 448-core cap
and have no additional user-imposed core-count cap; current Slurm limits,
including the shared SL4 CPU-minute budget, still apply.
Keep shared policy in `AGENTS.md` so Codex and Claude follow the same rules.

**Poll `squeue` no more often than about once every two minutes (120 seconds),**
including agent-issued checks and monitoring scripts. Batch relevant jobs in
one query, reuse recent timestamped results, and coordinate shared snapshots
across tasks. Poll less often when possible; do not replace `squeue` with
other Slurm status commands merely to check the same queue state faster.

**Never release or cancel the allocation hosting Codex or its remote runner.**
Do not shorten its walltime, resize away its host resources, or target it with
an automatic cleanup timer. This protection overrides completion cleanup and
the two-hour idle rule, and applies to Claude's allocation management too.
Identify the protected hosting job before releasing any worker allocation;
leave potentially hosting allocations untouched if that identity is unclear.
