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

Apply AGENTS.md's **Slurm traffic across agents and workflows** rules to all
scheduler activity, including submissions, step launches, cleanup and client
retries. Aim for roughly one controller request per two minutes across
coordinated tasks in steady operation; this is a soft target, with bounded
necessary startup/recovery/cleanup bursts. Share queue snapshots and launch
budgets, refresh filesystem demand between submissions, account for running,
pending and in-flight workers, and reuse long-lived allocations. The site's
rate-limit allowance is not a submission target. Reconcile in-flight jobs
before replacing a controller; documentation edits alone do not change an
already-running coordinator.

Normal allocation growth uses **one submission opportunity every five minutes
(300 seconds), not ten**. When multiple useful workers share a resource
profile, prefer a small demand-sized job array in that submission. Skip the
opportunity if existing capacity covers demand. Account for every array
member, including pending/in-flight members, in the ledger and resource
limits; ensure queue parsing, recovery and cleanup support array identities.
The soft RPC target is not a one-allocation-per-submission restriction.

**Never release or cancel the allocation hosting Codex or its remote runner.**
Do not shorten its walltime, resize away its host resources, or target it with
an automatic cleanup timer. This protection overrides completion cleanup and
the two-hour idle rule, and applies to Claude's allocation management too.
Identify the protected hosting job before releasing any worker allocation;
leave potentially hosting allocations untouched if that identity is unclear.

For SL4 CPU workers, follow AGENTS.md's **SL4 worker size, partition and walltime**
policy: large workers (roughly half a node or more, including the established
48-CPU profile) use `icelake`/`icelake-himem` for **two hours**. Small workers
prefer `sapphire` for its per-core performance, falling back to Ice Lake if
Sapphire admission is blocked or unsuitable. Use actual CPU/memory allocation
sizes, preserve healthy running work, and do not assume partition changes
bypass shared QoS limits. This adds no SL4 aggregate core cap.

For large checkpointable CPU campaigns, also follow AGENTS.md's **Building
useful SL4 capacity incrementally** section. The successful historical pattern
was a bounded shared task queue served by staggered, mostly 48-CPU/two-hour SL4
workers, reaching 688 running cores—not one large reservation. Let realistic
small core-time commitments admit as shared headroom opens; record each job,
respect polling/deadline rules, and release workers when no useful work remains.
These figures are examples, not a guaranteed quota. Use SL3 for time-critical
work when needed, within the combined 448-core SL3/SL2 cap.
