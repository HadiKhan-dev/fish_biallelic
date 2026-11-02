# Workflow mode

Each workflow command executes a cut of the same dependency graph used by `run`.
All jobs use the same configuration and shared checkpoint directory. Resource
options may differ by job; scientific settings and input order must match the
initialised run.

| Step | Form | Required published inputs |
|---|---|---|
| Initialise once | `haplotypes init cross.toml` | Configuration and source observations/templates |
| Evidence per contig | `haplotypes evidence cross.toml --contig chr1` | Initialisation |
| Relatedness gather | `haplotypes relatedness cross.toml` | Evidence from every configured contig |
| Reconstruct per contig | `haplotypes reconstruct cross.toml --contig chr1` | That contig's evidence; includes discovery through painting and refit |
| Pedigree gather | `haplotypes pedigree cross.toml` | Relatedness and reconstructed contigs |
| Phase per contig | `haplotypes phase cross.toml --contig chr1` | Pedigree and that contig's prerequisites; includes mapping and export |
| Reports gather | `haplotypes report cross.toml` | Available final records; after phase in the complete workflow |

Reconstruction can overlap the relatedness gather because its per-contig inputs do
not include relatedness. Pedigree is the barrier that needs both branches.

## Shared-directory protocol

`init` binds the resolved configuration, source content, sample/contig order and
realised seeds. A repeated identical initialisation is a no-op; a conflicting
binding is refused. Every other command requires initialisation and verifies the
binding. Simulation initialisation materialises all contigs before scatter jobs.

Tasks publish under derivation identities and use atomic claims. Duplicate
submissions of the same unit cannot replace a scientific publication with a
conflicting result. Scatter tasks write their own contig container; gather tasks
publish run-wide products. Invocation logs and run records remain distinct.

Workflow markers live in **CHECKPOINTS/markers**. With the default checkpoint
location this is `run/checkpoints/markers`, not `run/markers`. Global steps use
`init.done`, `relatedness.done`, `pedigree.done` and `report.done`; contig steps use
`evidence.CONTIG.done`, `reconstruct.CONTIG.done` and `phase.CONTIG.done`. The command
writes a marker only after validating the relevant products. If it dies between
publication and the marker, rerunning verifies the products and recreates the
marker. Never fabricate markers with `touch` as a substitute for running a step.

## Snakemake example

The complete [Snakefile](../examples/Snakefile) reads `cross.toml` directly so its
contig list and checkpoint root match the CLI. Copy it next to that configuration
and run Snakemake from that directory:

```sh
snakemake --snakefile Snakefile --cores 8
```

The example gives each scientific command its rule's `threads` value through
`--cores`. Adjust those resource requests for your machine or workflow executor.
No scheduler, account or machine-specific paths are embedded. The marker path is
quoted and configured explicitly. Use ordinary filename-safe contig names in this
example; the package's own record identities do not depend on such file naming.

The matching serial shell order for the five-contig example in
[running a cross](running.md) is:

```sh
haplotypes init cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes evidence cross.toml --contig "$c" --cores 8
done
haplotypes relatedness cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes reconstruct cross.toml --contig "$c" --cores 8
done
haplotypes pedigree cross.toml --cores 8
for c in chr1 chr2 chr3 chr4 chr5; do
    haplotypes phase cross.toml --contig "$c" --cores 8
done
haplotypes report cross.toml --cores 8
```

This is orchestration of supported commands; it does not add a scientific stage.
The release workflow-equivalence check compares these cuts with `run`, including
published records and files. The documentation build checks the example against
the current parser and the integration test's recorded result.
