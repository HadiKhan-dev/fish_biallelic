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

**Never release SL4-CPU allocations before they expire naturally.** The user
is not charged for SL4-CPU ("don't free up SL4 until they naturally expire, I
don't get charged for those", 7 October 2026). Keep every running SL4 worker
until its scheduler walltime ends, even when it is idle or its task is
finished, and give it useful authorized work meanwhile; no completion cleanup,
idle timer, shortened TimeLimit or scheduled cancellation may end it early.
This overrides the completion-release and two-hour idle rules for SL4-CPU
only (AGENTS.md: **Keep SL4 allocations until they expire naturally**). Other
service levels keep those rules, and submission pacing and the shared SL4
CPU-minute budget still apply.

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

**Whole-genome runs require at least a full node**, including validation,
merge-correctness checks, checkpoint resume tests, canaries, and delegated
checkers. Follow AGENTS.md's **Full-node allocations for whole-genome runs**:
for successive tests, use a reusable **76-CPU, single-node Ice Lake SL3-CPU
allocation for 12 hours**, or reuse suitable existing full-node capacity with
sufficient remaining time. This specific policy overrides general cheaper-tier
preferences and the partial-node SL4 worker profile. Verify actual allocated
resources and worker affinity, and propagate the full CPU budget through the
test harness and scientific workers so parallel stages use it. Never quietly
fall back to 16 CPUs or another partial-node allocation, even for a final check;
queue or coordinate full-node capacity instead. Preserve the aggregate paid
448-CPU cap, protected runner, and submission pacing. Smaller genuinely bounded
unit or chromosome checks remain permitted; whole-genome tests are not exempt.

**Keep managed SL4 allocations for known compatible follow-on work.** A task
finishing, or the ready queue briefly emptying, must not automatically release
an allocation that upcoming authorized work can use. Keep the worker/batch
session alive, coordinate the next task and expected readiness, and reuse the
same allocation across task boundaries and preparation/dependency gaps before
submitting replacements. For example, a 12-minute task in a two-hour, 48-CPU
allocation should leave the remaining 1 hour 48 minutes available for known
next tasks. This normal reuse does not require evidence of worsening cluster
contention. Respect remaining resources and scheduler expiry. Running SL4
allocations are kept until they expire naturally (above), so neither a lack
of follow-on work nor the two-hour idle limit releases them, and the protected
runner must never be released.

For SL4 CPU workers, follow AGENTS.md's **SL4 worker size, partition and walltime**
policy: **prefer `icelake-himem` for suitable Ice Lake requests**, especially
large workers (roughly half a node or more, including the established 48-CPU
profile), with **two-hour** walltimes. Use ordinary `icelake` as a documented
fallback when high-memory capacity is inaccessible, unsuitable, or current
admission evidence favors it. On 29 September 2026, `icelake-himem` had
`PriorityTier=2`, versus tier 1 for `icelake` and `sapphire`; partition tier is
considered before numerical job priority. Reuse shared observations and verify
these dated settings at useful decision points, not per submission.

Small workers may prefer `sapphire` when prompt admission and per-core
performance make it useful; when admission is constrained, prefer suitable
`icelake-himem` before ordinary Ice Lake. Request genuine CPU/memory needs,
preserve healthy running work and the protected runner, and reconcile any
pending replacements. Higher tier does not bypass the shared CPU-minute budget
or guarantee a start. Retain submission pacing and avoid duplicate requests;
this adds no SL4 aggregate core cap.

For large checkpointable CPU campaigns, also follow AGENTS.md's **Building
useful SL4 capacity incrementally** section. The successful historical pattern
was a bounded shared task queue served by staggered, mostly 48-CPU/two-hour SL4
workers, reaching 688 running cores—not one large reservation. Let realistic
small core-time commitments admit as shared headroom opens; record each job,
respect polling/deadline rules, and keep running workers until their natural
expiry rather than releasing them early.
These figures are examples, not a guaranteed quota. Use SL3 for time-critical
work when needed, within the combined 448-core SL3/SL2 cap.
