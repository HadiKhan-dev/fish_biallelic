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
`init.done`, `relatedness.done` and `pedigree.done`; contig steps use
`evidence.CONTIG.done`, `reconstruct.CONTIG.done` and `phase.CONTIG.done`. The command
writes a marker only after validating the relevant products. If it dies between
publication and the marker, rerunning verifies the products and recreates the
marker. Never fabricate markers with `touch` as a substitute for running a step.

`report` only reads checkpoints. It writes tables, `output_selection.json`,
invocation records and `report.done` in the output directory. The report marker
is an output completion signal, not a checkpoint publication. To republish from
an archived root, choose a writable output and scratch directory:

```sh
haplotypes report cross.toml --checkpoints /archive/cross/checkpoints --output republished --scratch republished/scratch --cores 8
```

Keep the original scientific configuration and numerical environment. Optional
output choices apply to this publication; pair reports require an existing pair
publication. The original checkpoint files and markers remain unchanged.

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

## Several nodes, one checkpoint root

For large runs, scatter chromosomes to separate nodes, then let one `run`
invocation finish the genome-wide stages. Use a shared filesystem for the
configuration, inputs, checkpoint root and output directory. For example, set
`[run] output` and `checkpoints` to shared absolute paths in `cross.toml`.
Initialise once, then produce evidence for every configured chromosome:

```sh
haplotypes init cross.toml --cores 8
haplotypes evidence cross.toml --contig chr1 --cores 8
haplotypes evidence cross.toml --contig chr2 --cores 8
haplotypes evidence cross.toml --contig chr3 --cores 8
haplotypes evidence cross.toml --contig chr4 --cores 8
haplotypes evidence cross.toml --contig chr5 --cores 8
```

Evidence commands may themselves be scheduled concurrently after `init` exits
successfully. Submit one reconstruction job per chromosome to your batch system;
these commands can execute on different allocated nodes at the same time:

```sh
haplotypes reconstruct cross.toml --contig chr1 --cores 8
haplotypes reconstruct cross.toml --contig chr2 --cores 8
haplotypes reconstruct cross.toml --contig chr3 --cores 8
haplotypes reconstruct cross.toml --contig chr4 --cores 8
haplotypes reconstruct cross.toml --contig chr5 --cores 8
```

Use each job's allocated CPU count instead of the illustrative eight cores. Start
long chromosomes early. Once every scatter job has exited, run the finishing
command on one allocated node:

```sh
haplotypes run cross.toml --cores 8
```

It reuses completed chromosome publications, runs relatedness if needed, and
finishes pedigree, phase, recombination, export and reports. A failed scatter can
be rerun on its chromosome or left for this finishing command to resume. The
relatedness gather may overlap reconstruction because reconstruction has no
relatedness dependency. Pedigree still needs both branches for all contigs.

Safety conditions:

- All jobs use the same fixed package sources, numerical-library policy, resolved
  configuration, inputs, maps, sample/contig order, realised seeds and tie settings.
  Resource limits may differ. Do not edit sources or inputs during the scatter.
- Initialise successfully before scattering; simulation init materialises every
  contig once. Each reconstruction requires its own published evidence.
- The shared filesystem must support coherent cross-node `flock` and atomic
  rename. Publications use per-derivation claims and append locks. Duplicate
  submissions reuse one publication, but still waste resources.
- Give each job its allocated process/thread budget and sufficient memory and
  scratch capacity. Scratch must be writable; it is disposable and never a
  checkpoint prerequisite.
- Wait for every scatter writer to exit before the final `run`. Run one finishing
  invocation and one report publication at a time for a given output directory.
  Do not remove claims or completion markers to force progress, and recover an
  abandoned claim only after its writer is confirmed terminal.

This uses the same dependency graph and identities as a single-node `run`; the
scheduler changes placement and timing, not scientific decisions.
