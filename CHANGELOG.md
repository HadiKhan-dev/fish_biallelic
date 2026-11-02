# Changelog

## 4.0.1 (2 November 2026)

Adds the manuscript describing release 4.0, rebuilt from scratch from the 4.0 methods
documentation and code, in [paper/](paper/README.md): the main text and Supplementary Information
as Markdown sources, the bibliography, figure and table generators, and a build script that writes
the PDF, HTML, DOCX and LaTeX versions. Figure 2 and Supplementary Tables S1 and S2 are drawn
directly from the [cost page](docs/developer/costs.md) and the
[numerical ledger](docs/developer/ties.md). Results that need the outputs of the 4.0 validation
matrix are marked as placeholders in the text.

The scientific code, settings, defaults and outputs are those of 4.0.0. The package's root
initializer carries the version number and is part of every stage's code identity, so 4.0.1 has
new code identities: resuming a run root started under 4.0.0 stops with a checkpoint mismatch
(exit status 2). Finish or resume 4.0.0 runs with a 4.0.0 checkout, and start 4.0.1 runs in a new
output root. For the same reason the workflow receipt (`docs/examples/workflow_receipt.json`) still
names the 4.0.0 sources until the workflow integration test is rerun and the receipt regenerated
with `docs/workflow_receipt.py`.

## 4.0.0 (2 November 2026)

The rebuild separates pure scientific models from runtime orchestration, typed
record publication and output formatting. It adds checked product schemas,
explicit randomness and tie policies, validated storage routes, workflow step
commands and generated documentation. `run CONFIG` still initialises a new run
automatically. Workflow steps require explicit `init` first.

Configurations migrate through `haplotypes migrate-config OLD.toml`: exact
supported translation or an explicit refusal. Unknown keys and removed metadata
are not ignored. A 4.0 run starts with a new checkpoint root; it cannot resume 3.5.
See the [migration guide](docs/user/migration.md) for commands and output changes.

Pairwise `relatedness/pairs.csv` is now optional and off by default (decision 0d).
Changing this output selection does not recompute families. Standalone export no
longer accepts `--vcf`; the stored released-site contract carries source alleles
and contig metadata. Pedigree tables render typed views with preserved text
encodings. Family numbers are canonical; phase records omit obsolete display and
decoder frames.

The [numerical ledger](docs/developer/ties.md#numerical-ledger) describes L1–L15:
unified Haldane fractions and reduction orders, stable duplicate exchange ties,
float64 chain tests, deterministic proof admission and family numbering,
content-based decision sequences, explicit tie settings, deterministic work-stop
witnesses, direct refit gains, map-based context distances, canonical powered
emissions, consistent relatedness move likelihoods and template kernels without
fast arithmetic reassociation, and unphased sample genotypes in ascending allele order.
These are declared numerical changes, not blanket
tolerances for differing scientific output. The owner approved ledger rows L1–L15
and their scoped bounds.

**Validation.** One matrix of 55 runs compares this release with 3.5: truth
simulations, TroMau, references and perturbations. When the release was committed,
30 runs had finished, all with exit status 0, and 29 had been scored. Founders,
paintings, refits, pedigree calls, phase and exports match 3.5. The differences
are declared ledger rows: floating-point drift of at most 1.5e-9 in pedigree
diagnostics (L2, L12), canonical family numbers (L6) and count-step diagnostics
(L7: 3.5 logged repeated rejected death and birth steps that 4.0 skips, so
join-death decision counts are lower by 1 to 28 per run). TroMau gives identical
results at 76 and at 56 cores. The ties report, which omitted discovery's and local
selection's blocks in earlier 4.0 candidates, matches 3.5 apart from those L7
counts. On the same hardware a run takes 1.29-1.36 times 3.5's wall time on TroMau
and 1.5-1.9 times on the simulations.

Retained cost exemptions are context-coded local-selection screening (0), ranked refits
(0b), bounded discovery/selection searches (0c), and optional pairwise IBD
(0d); see the [cost and accretion register](docs/developer/accretions.md).
Four worst-case costs above cubic in K are carried exactly as documented
exceptions (owner decision 0e, the treatment of 0c) and listed for review in 4.1;
the [cost table](docs/developer/costs.md) names them.

The reference platform is Linux x86_64, CPython 3.14.8, with exact runtime pins in
[requirements-tested.txt](requirements-tested.txt). The
[environment reference](docs/reference/environment.md) is generated from those
pins. Packaging requires Python 3.14 or later; the minimum dependency versions are
the tested ones. `llvmlite` and `threadpoolctl` are now declared runtime
dependencies, and `openpyxl` is no longer one. No GPU dependency is introduced.
