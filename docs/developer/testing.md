# Testing and scientific comparison

Validation proceeds from small deterministic units to realistic stage seams,
then composed products and release candidates. A passing unit test does not
establish stage identity; a good aggregate metric does not explain a changed
scientific call.

| Tier | Purpose |
|---|---|
| Unit/reference | Tiny independent calculations, exhaustive state cases, invariants and invalid inputs |
| Fast oracle | Small recorded seams exercising specific decisions and regressions |
| Stage | Realistic mechanism-selected fixtures, including near-cutoff controls on both sides |
| Composition | Connected stages using actual published products and checked adapters |
| Candidate | Supported command/workflow, portability, identity, costs and truth-based release gates |

The oracle guide beside the oracle tests in the source repository defines the immutable reference
format, extractors, comparison and decision register. Fixtures state what they
cover. Compare each field under its declared role: calls and structural state are
exact; continuous quantities use a reviewed, quantity-specific rule. Diagnostic
exclusions are explicit, never inferred from convenient key names.

## Decisions and numerical changes

A decision-register entry names its producer, operands, policy, threshold or tie
quantum, comparison direction and replay hook. Recompute the decision on both
reference and candidate operands. Include a near-boundary pass and fail, exact
cutoff behaviour and a wrong implementation that the fixture rejects.

For a [ledger normalisation](ties.md#numerical-ledger), retain the first divergent
operands, both decisions, margins, number of flips and downstream effects. A
missing fixture or unreviewed map is an open gate. Do not widen a tolerance to
cover an unexplained discrete change or report an unobserved count as zero.

Exercise explicit convention reversal separately from model priors. Force both
storage routes, cache states, worker counts and recycling paths where supported.
Test payload mutation and identity refusal, torn publication recovery, missing
products and copied-root bindings. Import tests verify that loading science has
no file/environment/runtime effects.

## Documentation checks

```sh
python docs/generate_docs.py --check
python tools/check_docs.py
```

The docs checker validates Markdown targets/anchors, schema keys and example
values, CLI commands/choices, cited parameter values, generated references,
environment reads, portability and carried-unit coverage. Negative controls live
under `docs/tests`. Workflow validation uses the actual
[Snakemake example](../examples/Snakefile), or its exact shell equivalent, with a
source/candidate receipt (`docs/examples/workflow_receipt.json`, projected from the
integration test by `python docs/workflow_receipt.py`). Release readiness requires
that receipt to match the frozen candidate. `python tools/check_docs.py --draft`
lists the receipt as a pending gate while sources still change; a draft check does
not satisfy the release gate.

Scope lint/type/test commands to the files being changed. Record commands,
source identity, fixture identities and outcomes so another developer can replay
the evidence. See [kernel declarations](kernels.md) for compilation-route audits.
