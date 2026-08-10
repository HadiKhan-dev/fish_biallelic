# Scientific coding task brief

Use this template at the start of a substantial task. Delete sections that are not relevant.

## Objective

What concrete result should exist when this task is finished?

## Biological question

What biological interpretation or hypothesis is the code intended to support? Which facts are known, inferred, or uncertain?

## Mathematical/statistical scope

Which model, likelihood, prior, objective, threshold, convergence rule, or numerical method is in scope? Which must remain unchanged?

## Supported execution path

Which entry point, configuration, dataset subset, stage, and checkpoint path matter?

## Acceptance criteria

- Required behaviour:
- Required scientific checks:
- Required numerical checks:
- Required runtime, peak-memory, I/O, and scaling bounds:

## Performance and resource scope

Is performance part of acceptance for this task? State the intended workload,
current baseline, suspected bottleneck, verified Slurm resources,
process/thread plan, memory budget, and the smallest representative benchmark.
Leave this section blank when performance is not material.

## Non-goals

Explicitly list tempting but unwanted work, for example:

- no general security or tamper-hardening;
- no repository-wide refactor;
- no new abstraction unless the requested change genuinely needs it;
- no full-pipeline run;
- no unrelated work outside the stated objective.

## Validation budget

Smallest representative test, maximum runtime, maximum CPUs/workers, and whether Slurm submission is authorized.

## Delegation

Identify any biological, mathematical, code-path, implementation, or validation questions worth delegating. State what each delegated agent must return and how the coordinator will reconcile the results. The repository does not impose a fixed concurrency or nesting limit; use the active orchestrator responsibly and preserve the task scope.

## Completion condition

Stop when the acceptance criteria pass. Report remaining uncertainty; do not add optional hardening afterward.
