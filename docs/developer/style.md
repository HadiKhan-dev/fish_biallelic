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

Every scientific constant belongs to the owning package's parameter module (for example, `science/<area>/parameters.py`). Its provenance states meaning, units and the reason for its value.
Methods prose names parameters; the [generated reference](../reference/parameters.md)
is the place for default values. Distinguish model, search and release controls
from storage or scheduling choices. A byte-valued search budget can still affect
science and must not follow free memory.

Use one declared expression for each numerical quantity. Preserve model meaning,
thresholds, search order and named tie mechanisms. Mathematically equivalent
reorderings (another summation or reduction order, vectorisation, fused
multiply-add) are acceptable and need not preserve bits. What must stay explicit
is every decision: the dtype at a decision, exact-zero tests, and named tie
mechanisms with their tolerances. When such a change moves a decision across a
threshold, report how many decisions flipped and by what margins. Where a
decision must not depend on sample order or labels, compute its operands with an
order-independent (exactly rounded) sum. The [tie guide](ties.md) documents
current numerical conventions. Numba kernels are compiled without fastmath,
which would also change NaN and infinity semantics.

Use [named streams](ties.md#random-streams) with semantic keys and explicit
conventions. No process-global random state or environment-selected scientific
behaviour is allowed. A diagnostic sink must not feed a later decision. Keep
scientific errors typed; let the interface translate input, checkpoint and runtime
failures into user-facing messages.

Science performs no file, clock, logging or warning effects. Stage adapters own
publication and codec conversion. Runtime records own timings and resource
measurements. Results own read access; outputs own textual formatting. Declare
cross-science APIs in the importing package rather than reaching into internals.
