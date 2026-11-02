# User guide

Haplotype Reconstruction reconstructs founder haplotypes from low-coverage reads,
paints each sample as a diploid founder mosaic, infers relationships among the
sequenced samples, and estimates phase and recombination conditional on those
relationships. The input is an indexed multi-sample VCF or BCF and a genetic map
or a stated fallback rate. Sample names are identifiers; the analysis reads no
cohort, generation, sex or parentage metadata.

Start with [installation](installation.md) and the [synthetic quick start](quickstart.md).
Then use [running a cross](running.md) to prepare real observations and interpret
resume/status. [Workflow mode](workflow.md) distributes contigs through a shared
run directory. Read the [limitations](limitations.md) before interpreting parent
calls or recombination rates.

- [Output dictionary](outputs.md): exported genotypes, tables, diagnostics and evaluation metrics.
- [Settings reference](../reference/settings.md): accepted TOML keys and their current defaults.
- [Command reference](../reference/commands.md): every command and option.
- [Migration from 3.5](migration.md): new runs, explicit configuration translation and output changes.
- [Methods](../methods/index.md): models, decision rules and approximations by semantic unit.

An unresolved parent or missing allele is a result, not an invitation to fill it
from a sample label. `relatedness/pairs.csv` is optional and off by default; the
relatedness family model still runs. Simulation evaluation uses generated truth;
it does not turn names or cohort labels in a real cross into known relationships.
