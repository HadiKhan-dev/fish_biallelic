# Changelog

## 4.2.0 (3 November 2026)

This release covers the K = 30 review that 4.0 deferred, and the smaller items left after 4.1. Science changes are
listed first. Every other change leaves released outputs scientifically identical to 4.1.

**Founder-count refits evaluate every hypothesis.**
- The assembly joins' duplicate-seed births, and refinement's deletion and addition rounds, now refit every eligible
  candidate. Before, they refit only the nine best by an unrefitted proxy (`ranked_refits`, cost exemption 0b).
- On the 4.0 release matrix the cap left 1,597 deletion candidates and 111 addition seeds unfitted. Replays on saved
  inputs found two rounds where an omitted candidate fitted better than every admitted one. In both, the count
  verdict was unchanged:
  - a TroMau deletion at K = 10, better by 54.9 nats;
  - an addition in the K = 30 simulation, better by 103.1 nats.
- The decision rule is unchanged: best endpoint F, its tie class and content-smallest member, the symmetric null,
  the bootstrap streams and the shadow families.
- Each count round now fits O(K) candidates, O(N L K³). A pass has at most one more deletion round than it accepts
  deletions, so its worst case is quartic in K. Saved runs had one to five rounds per pass.
- The zero-gain cleanup before refinement's deletion rounds is removed. It fired in none of 54,123 saved component
  refinements, and the ordinary count rule now decides every deletion.
- The tie census loses the shortlist and cleanup kinds.
- Count-rule settings and count-decision records are schema 2.

**Pedigree exclusion evidence is stored as columns.** It used to be one object per screened dyad, about 4 million
at N = 2,500. On a saved N = 2,503 decision seam:
- every rendered table is unchanged;
- the decision step takes 395 s instead of 433 s;
- the seam with its reports takes 405 s instead of 476 s;
- peak memory is 9.7 GB instead of 12.6 GB.

The pedigree product is schema 3.

**Smaller changes.**
- Painting's entry points require the tie-convention reversal explicitly.
- Label continuity groups tie peers once per iteration, so peer lookup is linear in the candidate count.
- Discovery's reader checks each block once on first read.
- The serial re-export route is removed. `haplotypes export` already exported contigs in parallel.
- Duplicated code is removed from phase CAL2, founder identity flags and the painting support tables. Their output
  is byte-identical.
- An unused painting memory setting is removed.
- The example VCF header captures are now `docs/examples/*.header.vcf.txt` and ship with the repository.
- Type declarations are corrected (`JoinSummaryView`, `count_vector_classes`).
- The environment reference lists the thread-limit variables the runtime fixes.

**Reviewed and not changed.**
- **Discovery's bounded searches (exemption 0c):**
  - the replacement starts (48/96) and the archive beam (6/12) bind at every K;
  - they are part of the search's design and need a redesign, not a larger constant;
  - exact gauge cuts certified by component structure are implemented and validated but held back. On their own
    they move three TroMau blocks, two of them to panels the search's own criterion scores lower.
- **Local selection's backstops, bank budgets, context snaps and the LS10/LS17 caps (exemption 0c):** each needs a
  new search or stopping contract.
- **The joins' structural screen (AJ17):** its adaptive route has a cubic analytic bound.
- **Refinement's canonical window fixing (RS05/RS07):** no work stop in any saved run.
- **Painting label continuity and founder-refit tie classes (PR17, FR04):**
  - their worst cases remain above cubic;
  - a refit reorganisation with identical decisions was prototyped, but it does not lower the bound, so it was not
    adopted.

The [cost page](docs/developer/costs.md) and [accretion register](docs/developer/accretions.md) record each finding.

**4.1 validation, completed.** All 39 cases of the 4.1 release matrix finished. Every released output equals 4.0's,
and the ties reports equal 4.0's corrected ones. Wall time, 4.1 over 4.0: median 0.64.

**Validation.** The fast test tier passes (2,914 tests). An independent review of the candidate found no defect.
The release matrix (39 runs: truth simulations, TroMau, references and perturbations) was compared with 4.1:
- **Released outputs identical (32 runs, including TroMau under reversed tie conventions):** only the ties report
  differs (the removed census kinds). In nine of them the assembly search tables and count diagnostics also list fewer
  search events.
- **K = 30 simulation, scored against truth:**
  - Founders improve. One chromosome now releases the true 30 founder rows (4.1 released 32), with called-allele
    errors falling from 1,431 to 603. Founder errors also fall on two other chromosomes.
  - Genotype errors fall from 1,387 to 1,376 and the phased share rises from 0.966 to 0.967.
  - Switch errors move from 30 to 31.
  - Pedigree recovery stays weak in both releases: correct parent states 23 to 24, correct edges 6 to 4.
- **TroMau, the reference run and its five perturbations:**
  - chr3 assembles into one eight-founder component, like 20 of the other 21 chromosomes; 4.1 released three
    components.
  - chr3's recombination junction loci fall from 5 to 3 (none flagged).
  - The primary (Tier B) pedigree is unchanged.
  - Under reversed tie conventions chr3 keeps three components in both releases, so this structure is
    convention-sensitive.
  - Run time on the same node type is unchanged (6,278 s and 6,146 s).
- **Cost of evaluating every count candidate at K = 30:** the K = 30 simulation takes 52.6 minutes against 36.2 for
  4.1, measured on two identical 76-core nodes, with the same peak memory. Both runs reproduce their matrix results
  exactly.

Code identities change, so 4.2 cannot resume 4.1 checkpoints. Start a new checkpoint root.

## 4.1.0 (3 November 2026)

A speed release: scheduling, transport and start-up change; no model, parameter,
threshold, tie rule or output format does. Released outputs are meant to be
scientifically identical to 4.0's. It includes 4.0.1's manuscript
([paper/](paper/README.md)), whose methods are those of this release.

**Start-up.** Kernels are compiled as tasks of the run's own graph, beside the
stages, instead of all of them before the first stage; a stage that meets a kernel
still being compiled waits for that kernel only, and one process at a time writes
each kernel's cache entry. Initialising an observation (VCF/BCF) source runs in the
calling process, without a numerical runtime of its own. The worker server starts
while the graph is planned. On a cold cache the first stage starts within seconds
of the command instead of after the whole warm-up.

**Hand-offs.** A contig's decoded discovery and evidence inputs stay in the worker
that runs its selection and assembly chain, and an expanding task's continuation
stays in that worker instead of travelling through the coordinator. Record decoding
and code-identity hashing are cached per process. A task's first CPU share no
longer borrows from peers that have not taken theirs.

**Tail.** Report tasks run on any free worker; each contig's tie reader follows the
completed stages rather than the exports; recombination runs before export; the
default and mirror phase solves release their samples as one batch; the phased-sample
VCF writer formats a chunk of cells at once, chunks bounded by cells.

**Performance.** On the same nodes (112-core Sapphire Rapids, cold cache, a
116-sample, three-chromosome subset of a real cross), a whole run takes 562-589 s
against 733-777 s for 3.5 (4.0 took 1,265 s where 3.5 took 780 s), and 791-817 s
against 988-1,015 s on 76-core Ice Lake. Whole-run CPU occupancy is about 49% (3.5: 44-46%).
The longest remaining serial stretches are the final refinement level's count pass
on the largest chromosome and each chromosome's evidence reading.

Run records gain each task's `finished_seconds`, and kernel compilations appear as
operational tasks (`kernels/compile/run/<module>:<function>`). Code identities
change with the engine sources, so a 4.1 run cannot resume 4.0 checkpoints: start a
new checkpoint root.

**Validation.** The fast test tier passes. Released outputs on the reference cross
matched between successive candidates. The release matrix (truth simulations,
TroMau, references and perturbations; the hardware-variant rows of 4.0 are dropped)
compares this release with 4.0 and was running when the release was committed. The
workflow receipt (`docs/examples/workflow_receipt.json`) still names the 4.0.0
sources until the workflow integration test is rerun on 4.1.

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
