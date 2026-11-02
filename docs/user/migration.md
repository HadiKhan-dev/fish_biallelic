# Migrating from 3.5 to 4.0

**Start a new run and checkpoint root. 4.0 does not resume 3.5 records.** The rebuild
preserves the carried models and release rules, with the explicitly documented
numerical conventions and output changes below. A translated TOML file is not a
checkpoint conversion.

## Translate the configuration

Keep the original file and write the translated configuration separately:

```sh
haplotypes migrate-config cross35.toml -o cross40.toml
```

Without `-o`, the translated TOML and its per-key audit comments are printed to
standard output:

```sh
haplotypes migrate-config cross35.toml
```

The translator validates the entire result before writing. It either preserves a
supported value under an explicit translation or refuses with a reason; unknown
keys are not silently ignored. Review the audit comments and set `[run] output`
and `[run] checkpoints` to fresh locations before initialising:

```sh
haplotypes run cross40.toml --cores 8
```

Paths remain relative to the invocation's working directory. Moving the translated
TOML alone does not change their resolution. The generated [settings reference](../reference/settings.md)
and [command reference](../reference/commands.md) describe the accepted 4.0 schema.

| 3.5 input or convention | 4.0 action |
|---|---|
| Omitted/version-3 configuration version | Translator writes `version = 4` |
| Carried settings | Same keys and values are kept and listed in the audit |
| Empty `execution.process_contigs` | Explicitly dropped; scatter work uses workflow step commands |
| Nonempty `execution.process_contigs` | Refused; rewrite the orchestration with `evidence`, `reconstruct` and `phase` |
| Empty `execution.batch_queue` | Explicitly dropped; helper queues are replaced by workflow mode |
| Nonempty `execution.batch_queue` | Refused; use a workflow manager |
| `aulstu` preset | Refused; choose `generic` and specify the desired contigs yourself |
| Sample metadata, eligibility or parent-label sections | Refused; the pipeline reads no sample metadata |
| Unrecognised setting or unsupported value | Refused with the field and reason |

The two presets are `generic` and `tromau`. A preset does not infer known parentage
from sample names. Translation does not add metadata-based eligibility or adjust
scientific values to force a configuration to load.

## Commands and workflow mode

`run CONFIG` initialises a new run automatically, preserving the single-command
entry point. Explicit `init` binds a run before separate workflow step commands.
`run` and the scatter/gather commands execute the same model graph. The [Snakemake example](workflow.md) replaces
shards and the helper batch queue; `batch-worker`, `--process-contigs` and
`--batch-queue` are not 4.0 commands/options. Clean stop points and `--serial` remain
available as graph cuts.

Standalone `haplotypes export RUN DEST` **has no `--vcf` option**. It formats checked
ReleasedSites, FounderRelease and PhasedSamples records through the results reader,
including source REF/ALT and contig metadata. The original input file need not be
present for re-export. `--vcf` still belongs to input-consuming commands such as
`run` and `templates`; its removal is specific to standalone export.

`haplotypes recombination RUN` remains available. It uses the recorded final phase
and recombination binding. Once a contig has a mapped publication, changing its
recombination settings is refused, including in a copy of a completed run: copying
preserves the binding. Alternative settings need an **independent checkpoint root**
and valid prerequisites for that analysis. Copying a run or renaming an output
directory does not create an independent scientific identity.

## Pairwise IBD is an optional output

`relatedness/pairs.csv` is **off by default**. To retain the 3.5 pair report, explicitly
set:

```toml
[relatedness]
pairwise_ibd = true
```

The translator adds `pairwise_ibd = false` when the old configuration omits it and
explains that change in its audit. Read that note before assuming that all former
files will be emitted.

The pair chain costs O(N² L); family fitting does not require it. Selection is
persisted per invocation in `outputs/selection` and stays outside every scientific
identity. Switching the flag does not recompute families. When reports are written
with it off, a stale pairs CSV is removed even if the pair record remains in the
checkpoint store. Use the desired selection on the report invocation as well.

## Outputs and record consumers

Founder/sample VCF meanings, missing encodings, FILTER text, phase-set rules and
report cell encodings are preserved. Tables are now rendered from typed records.
The output dictionary includes relatedness tables, painting label-continuity/model
tables and the sibling-guard report; their omission from an older guide was not
an absence of those outputs.

| Item | 4.0 contract |
|---|---|
| Relatedness family numbers | Canonical dense ranks ordered by the family's smallest member (L6); match historical families by membership, not a former sparse integer |
| Assembly `source_decision` names | Deterministic names from the decision sequence (L7), not object identity |
| Phase record internals | `displayed_frame` and `decoder_frame` are not stored public products; use the released phase and ambiguity fields |
| Product/checkpoint schema | Typed, versioned and derivation-checked; no direct 3.5 pickle/checkpoint reuse |
| Export indexing | VCF chooses TBI/CSI by the last **written** position; BCF uses CSI |
| Sample ordering | Canonical name order in records, input order in published sample outputs |
| Standalone export destination | New or empty; re-export does not replace the original run's export record |

Use the supported [results readers](outputs.md#results-readers) instead of reading
private checkpoint entries or internal Python objects. The absence of a product in
a partial run is explicit; it is not reconstructed from a completion marker.

## Identity and numerical conventions

Input identity binds consumed content. Merely touching a file does not change it;
changing consumed AD, fallback GT, REF/ALT, contig length or loader drops does.
Scientific parameters, code dependencies, upstream identities and numerical-library
policy also enter derivation identities. Resource settings and optional report
selection do not. Completion markers describe verified publications; they never
replace their verification.

The [numerical ledger](../developer/ties.md#numerical-ledger) is narrow. It
replaces arithmetic that 3.5 computed in several forms, or in an order chosen by
workers, memory or object identity, with one declared form:

- L1: one Haldane recombination fraction, computed with `expm1`;
- L2: declared reduction orders, independent of worker counts and memory;
- L3: a stable content order for exact-duplicate exchanges;
- L4: family phase's chain test in float64, with float32 only for storage;
- L5: the shadow-domain proof attempted by a dimension rule, not by free memory;
- L6 and L7: canonical family numbers and decision-sequence names;
- L8: tie reversal and the tie census passed as settings, not environment variables;
- L9: a deterministic refinement work witness;
- L10: one direct two-row gain in founder refit;
- L11: the selection context refit's distances through the map;
- L12: one form of the pedigree's powered emissions;
- L13: one family log-likelihood function for relatedness moves;
- L14: the template linker without fast-math reassociation (only when templates
  are rebuilt);
- L15: unphased sample genotypes written in ascending allele order (`0/1`, never
  `1/0`), so their text no longer follows an arbitrary internal order.

No ledger entry changes a model, threshold, candidate set, search order or budget.
A cut-off crossing is reported with its margins and downstream effect rather than
hidden by a tolerance. The [changelog](../../CHANGELOG.md) records measured release
flip counts from the final validation receipts; an unmeasured count is not zero.

## Dependencies and cost exceptions

Use the release's tested Linux/Python dependency set rather than assuming the old
environment will produce the same numerical identity. The installed metadata and
`requirements-tested.txt` record the final envelope. The package does not require
a GPU. Declared dependency removals are listed in the changelog after the packaging
import audit.

The accepted 0, 0b and 0c search/cost exceptions, and the four above-cubic
worst cases of decision 0e, remain scientific limitations in 4.0, not performance
fixes applied during migration. See [limitations](limitations.md)
and the [cost table](../developer/costs.md), including K = 30. The optional pair
output is the explicit 0d output-contract change.
