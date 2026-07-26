# Standalone metadata-aware Tropheops V7-margin pedigree

This package is the retained, reviewable form of the selected Tropheops
pedigree analysis. It is deliberately separate from the linear production
pipeline because it uses biological-design metadata and an explicit set of
G0-to-F1 seed assignments. It does not search old `results_*` directories and
it does not treat file names, aliases, cohort names, or prior inferred
pedigrees as truth.

Run `python -m pedigree_v7 --help` for the complete command line interface.

## Scientific model

The compatibility mode preserves the selected V7-margin implementation:

1. Read the pre-pedigree T08 painting and the T10-carried founder block for
   each of the 22 compatibility contigs. T10's corrected painting is not used,
   avoiding feedback from the old inferred pedigree.
2. Use every retained founder-informative marker with a PL genotype likelihood
   in the BCF.
3. Match locally IBS-equivalent reconstructed founder haplotypes before
   building probabilistic G0 tracks.
4. Reconstruct each F1's two homologs by parental origin from its explicitly
   supplied G0 pair, unordered G0 tracks, and the F1 genotype likelihoods. This
   sitewise reconstruction is invariant to arbitrary local swaps of either
   G0's painted tracks.
5. Score every eligible F1 father/mother pair for each F2 with a linked
   block-composite transmission HMM. The seven original error/block/smoothing
   variants are retained.
6. Compare zero-observed-parent, father-only, mother-only, and two-parent
   states with candidate-count correction and the three original state-prior
   scenarios.
7. Aggregate two-parent identity evidence per chromosome using 90% equally
   spaced rank utility and 10% bounded soft likelihood-margin evidence. The
   margin is square-root block tempered and mixed with 1% uniform chromosome
   contamination.
8. Report chromosome bootstrap, leave-one-chromosome, and seven-variant
   stability. Tier A is the original strict V7 policy. Tier B requires a
   bootstrap majority, at least 5/7 variants, at least 18/22 leave-one-contig
   analyses, compatible parent-count evidence, and joint/marginal identity
   agreement.

The numerical transmission kernels were copied unchanged from the validated
exploratory implementation. The package reorganises I/O and makes the design
and seeds explicit; it does not change the likelihood, utility, bootstrap,
leave-one-contig, or tier decision rules.

## Required inputs

- `--bcf`: the indexed multi-sample biallelic BCF/VCF containing `PL`.
- `--metadata`: the Excel metafile, sheet `main_data`, with `primary_ID`,
  `alias_ID`, `santos_ID`, `generation`, and `sex` fields.
- `--checkpoint-dir`: the pipeline checkpoint directory containing
  `T08_viterbi_painting` and `T10_phase_correction` contig checkpoints. It is
  required for both new scoring and external-cache reuse because all 44 source
  checkpoint identities are validated before a cache is accepted.
- `--g0-seeds`: an explicitly named F1-to-G0 pair CSV. This is never inferred
  from an old output directory.
- `--output-dir`: a new or resumable analysis directory.

For an audit/re-aggregation of existing complete V7 contig caches, pass
`--cache-dir`. The BCF, metadata, checkpoint directory, candidate design, and
seed file must still be supplied so their provenance can be checked against
that cache's manifest.

### Eligibility in exact compatibility mode

Generation and sex metadata determine *eligibility*, not parentage:

- the sequenced G0 cohort must contain 1 male and 3 females;
- the F1 cohort must contain 8 males and 8 females;
- the F2 cohort must contain 96 individuals;
- every G0 male-by-female pair is eligible for an F1 seed;
- every F1 male-by-female pair is eligible for F2 inference.

The package fails rather than silently changing this design. A different cross
requires an explicit scientific design extension and validation; editing the
expected counts merely to make another dataset run would not be valid.

### Explicit G0 seed CSV

The preferred compact schema uses sample IDs, not aliases or row positions:

```csv
child,father,mother,seed_basis,seed_status,report_parent_edges
F1_SAMPLE_1,G0_MALE_1,G0_FEMALE_1,"stable inferred seed; not breeding-record truth",exact_stable_inferred,true
```

There must be exactly one row for every eligible F1, all three sample IDs must
occur in the metadata/BCF, every pair must satisfy the design, and
`seed_basis` must be non-empty. `seed_status` must be one of:

- `exact_stable_inferred`: an inferred pair that was stable in its source
  analysis but is not breeding-record truth;
- `documented_breeding_record`: a relationship supported by independently
  documented breeding records; this label is supplied by the user and is not
  established by this package;
- `computational_seed_only`: a pair used only to reconstruct homologs.

`report_parent_edges` is a strict `true`/`false` field. It may be true for the
first two statuses, but must be false for `computational_seed_only`. A false
value retains the seed for scoring while suppressing its G0-to-F1 edges in the
reported pedigree. The historical V5 assignment schema is also accepted for
exact compatibility when it contains `child_index`,
`joint_MAP_father_index`, `joint_MAP_mother_index`, and `exact_pair_stable`;
every stability value must parse strictly as true. Supplying either schema
does not turn an inferred pair into ground truth.

The retained Tropheops seed input is:

`results_tropheops_withFounders/Tropheops_pedigree_handoff/F1_G0_seed_assignments.csv`

It records all 16 pairs as `exact_stable_inferred` with reportable edges. These
are fixed inferred inputs, not independently confirmed parentage.

## Example

```bash
python -m pedigree_v7 --bcf fish_vcf_restriped/AcTm.biallelic.bcf.gz --metadata fish_vcf_restriped/X_AcTm_metadata.xlsx --checkpoint-dir .pipeline_checkpoints_tropheops_withFounders --g0-seeds results_tropheops_withFounders/Tropheops_pedigree_handoff/F1_G0_seed_assignments.csv --output-dir results_tropheops_withFounders/v7_margin_reproducible --threads 112 --bcf-threads 4
```

The example's 112 Numba threads are appropriate only inside a verified
112-CPU allocation. The default is one thread. The implementation uses one
process and a bounded Numba thread pool, so the process-by-thread product is
the requested `--threads`; BLAS/OpenMP oversubscription remains capped by
`thread_config`. BCF reading uses the independently bounded `--bcf-threads`.

A complete real-data run reads the BCF and two checkpoints for each chromosome,
then performs CPU-bound Numba HMM scoring. It may take tens of minutes to hours
and produces compressed per-contig caches. Use `--resume` after an interrupted
run. `--contigs ... --diagnostics-only` is the smallest representative scoring
benchmark and does not attempt incomplete aggregation.

## Cache provenance and resume safety

Each package-owned `contig_cache` has a `cache_manifest.json`. External caches
are accepted only when their manifest exactly matches the scoring-model
revision, scientific settings, ordered BCF samples, normalized metadata,
candidate arrays, selected local G0 pairs, and normalized seed content. The
seed mapping and provenance are content-hashed, so changing a seed invalidates
both cache reuse and output-directory resume.

The BCF, any adjacent BCF/VCF index, metadata workbook, and all 44 checkpoint
files are identified by resolved path, byte size, and nanosecond modification
time. This is a practical stale-cache guard, not a cryptographic content hash
of those large source files. A byte-for-byte replacement preserving all three
identity fields would not be detected.

Pre-package exploratory caches lack this manifest and are rejected by default.
`--allow-legacy-cache-without-manifest` is an explicit unsafe compatibility
override for one-off historical parity checks; it cannot verify which BCF,
sample order, design, seeds, checkpoints, or settings produced those scores.
Do not use it for a production rerun.

## Outputs

The output directory contains:

- `F1_G0_seed_assignments.csv`: normalized seed identities and provenance;
- `F2_parent_assignments_v7_margin.csv`: all leading hypotheses and Tier A/B
  flags;
- `F2_candidate_pair_evidence_v7_margin.csv`: all conditional pair evidence;
  Pair-level fields in pedigree rows are explicitly labelled conditional when
  the selected parent-count state is not `two_parent`;
- `F2_parent_count_states_v7.csv`: missing-parent-state sensitivity;
- `pedigree_v7_margin_best_estimate.{csv,fam}`: leading hypotheses, including
  uncertain calls;
- `pedigree_v7_margin_tier_A.{csv,fam}`: strict stability-qualified calls;
- `pedigree_v7_margin_tier_B.{csv,fam}`: Tier B-or-better calls;
- matching `_edges.csv` files, `settings.json`, `summary.json`, and
  `SCIENTIFIC_CAVEATS.txt`, and `expected_cache_manifest.json`;
- `contig_cache/v7_<contig>.npz` for a new scoring run.

The `.fam` files use `Parent1` as PID (eligible male/father) and `Parent2` as
MID (eligible female/mother), PLINK sex codes 1/2/0, and phenotype `-9`.

## Interpretation and limitations

There is no independent individual-level trio ground truth for this real
cross. Do not describe Tier A or Tier B as accuracy, a posterior error rate, or
confirmed breeding-record parentage. Chromosome bootstrap fractions and
normalised parent-state composite weights are internal stability diagnostics,
not calibrated biological probabilities.

The model supports zero, one, or two observed F1 parents for an F2, but the G0
seed file still supplies one computational pair for every F1 so that its
parent-of-origin homologs can be reconstructed. Those seeds may be inferred
and must retain their stated provenance. The compatibility design assumes the
metadata generation and sex fields are correct eligibility annotations. If
those annotations or the candidate set are incomplete, the resulting ranking
is conditional on that incompleteness.

The linked G0-to-F1 HMM is retained only as a diagnostic because long-range G0
track phase is not established. It does not contribute to primary F2 scoring.
No fixed family count or family-prevalence prior is imposed.
