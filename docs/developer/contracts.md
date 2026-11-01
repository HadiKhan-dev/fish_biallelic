# Contracts and products

A contract states what an array means, not just how many dimensions it has.
Boundary records declare named axes, dtype, units, field role, ordering, missing
encodings, ownership, mutability and view lifetime. Published products add a
schema version. The [generated record reference](../reference/records.md) extracts
these declarations from the current code; the [output dictionary](../user/outputs.md)
defines their public files and reader methods.

## Axes and units

The run binds configured contig order and canonical sample order. Store the input
sample order separately for external tables/VCFs. Founder rows are local to their
panel/component; a row number is not a global biological identifier. Record every
permutation and apply it to all arrays sharing that axis.

Positions in files are one-based bp where stated; internal interval contracts
state closed or half-open endpoints explicitly. Genetic distances are Morgans
internally, while map files use cM. The model map includes fallback tails. The
measured-coordinate view clips support to measured regions unless the fallback
rate is explicitly measured. Likelihood/log-score units are probabilities or nats
as declared, never inferred from a field name.

## Field roles

| Role | Meaning for validation |
|---|---|
| Input | Consumed observations or declared upstream evidence |
| State | A value that can influence later search, admission or release |
| Call | A discrete scientific output, including missingness and ambiguity |
| Report | A published interpretation of state/calls with its own format contract |
| Diagnostic | Work, numerical or operational evidence that does not steer science |
| Derived view | Reconstructible representation, with an explicit lifetime and storage policy |

A counter that determines whether a scientific budget binds is state, even if it
looks like a diagnostic. A proof-status field can be diagnostic only when the
specified decision is already independent of it. The oracle map declares exact
roles and comparison rules per field; a generic name-based exclusion is unsafe.

## ObservationSource and readers

Input adapters expose canonical observations and an identity for the consumed
payload. Export-specific allele/metadata access is an explicit source method used
to construct ReleasedSites; output formatting later reads that stored contract
through results. Original input files are not required for standalone re-export.
Simulation exposes the same observation contract while truth remains separate.

Results requests name the expected derivation, schema and decoder. A partial run
can return absence for a product, but a corrupt or incompatible present product
must raise. Selective item reads and windows verify the same envelope as complete
reads. Decode typed nested records without inventing omitted state fields.

## Ownership and lifetimes

Shared inputs are read-only after publication. Mutable workspaces belong to one
solve or task. A leased window or reconstructed message view can expire at the
next refill/read; its contract states exactly which operation invalidates it.
Parallel consumers receive private readers or copied flanks. Dense and windowed
views may have different lifetimes while carrying exactly equal values.

For a new field, classify its role, document its axes and missing values, update
the schema/adapter deliberately and add a comparison fixture. Never accept an old
checkpoint by filling an unknown scientific field with a convenient default.
