# Code style

Give each module one reason to exist and state it in the opening docstring. Name
the semantic units it owns, its allowed dependencies and the corresponding
[methods section](../methods/index.md). Functions should express one named
transition; split orchestration, numerical kernels, typed diagnostics and output
formatting at real boundaries.

Use frozen dataclasses for boundary records and explicit parameter records.
Document shape, dtype, axis order, units, missing values, ownership and view
lifetime. A frozen record does not freeze its array buffers: validation/publication
must enforce the required read-only contract. Use descriptive domain names from
the [glossary](glossary.md); short indices are appropriate inside a documented
kernel, not at a public boundary.

Every scientific constant belongs to the owning package's parameter module (`science/<area>/parameters.py`) with provenance.
Methods prose names parameters; the [generated reference](../reference/parameters.md)
is the place for default values. Distinguish model, search and release controls
from storage or scheduling choices. A byte-valued search budget can still affect
science and must not follow free memory.

Use one declared expression for each numerical quantity. Preserve reduction
order, dtype at a decision, exact-zero tests and named tie mechanisms. Mathematical
equivalence does not establish floating-point identity. Record an approved
normalisation through the [ledger](ties.md#numerical-ledger); do not silently
replace a recurrence, summation or tolerance. Compiler fast arithmetic
reassociation is disabled.

Use [named streams](ties.md#random-streams) with semantic keys and explicit
conventions. No process-global random state or environment-selected scientific
behaviour is allowed. A diagnostic sink must not feed a later decision. Keep
scientific errors typed; let the interface translate input, checkpoint and runtime
failures into user-facing messages.

Science performs no file, clock, logging or warning effects. Stage adapters own
publication and codec conversion. Runtime records own timings and resource
measurements. Results own read access; outputs own textual formatting. Declare
cross-science APIs in the importing package rather than reaching into internals.

Run Ruff on changed files, the relevant type checks and meaningful unit/oracle
checks. Keep generated references byte-identical to current source and make the
documentation build with zero warnings. Tests should exercise competing decisions,
boundary sides, storage routes and scientific invariants rather than repeat the
implementation line by line.
