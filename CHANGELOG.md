# Changelog

## 4.0.0 — release candidate

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

**Validation status:** measured release-wide flip counts and first-divergence
receipts have not yet been incorporated into this candidate changelog. The
candidate remains subject to that publication gate; absent counts do not mean
zero flips. Each final ledger row must link its measured count, margin and
downstream effect before this section becomes a release record.

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
