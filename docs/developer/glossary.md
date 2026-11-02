# Glossary

| Term | Meaning |
|---|---|
| Founder haplotype | An inferred ancestral allele row; its label is local to the relevant component |
| Diploid founder | A simulated individual carrying two founder haplotypes; not synonymous with one inferred row |
| Haplotype | One ordered allele sequence along a chromosome or component |
| Panel | Founder rows over a declared site axis, including explicit unknown calls |
| Prepared panel | A panel plus the coded local evidence/layout needed by assembly |
| Block | A local discovery/selection interval with its own panel |
| Component | An assembled span of blocks and their founder paths |
| Junction | A boundary at which paths may join/change; a reported map junction is a separate diagnostic concept |
| Painting | A sample’s copying-model assignment/posterior over founder trajectories and background |
| Bin | A consecutive site group with declared bp edges and a model-specific aggregation rule |
| Track | One displayed or latent homologue path; local track order need not identify a biological homologue |
| Edge | A released directed parent-to-child relation, conditioning one child slot |
| Slot | A parent/homologue position in a record; it carries no sex label |
| Phase set | A span with released relative homologue orientation, defined by components and explicit breaks |
| Meiosis chain | Per-bin released parental-homologue source, support and reset state for one parent edge |
| M0 / M1 / M2 | No, one or two observed-parent states; unobserved slots use a null source model |
| Candidate panel | The score-bounded set of M2 configurations that receives exact scoring |
| Complete view | An acyclic best-guess pedigree view, distinct from released support tiers |
| Tier A / Tier B | Separate support/consistency release thresholds, with explicit orientation and guard rules |
| Count-unverified | A bounded proof domain was not fully verified; interpretation depends on the owning contract |
| Unresolved search | A search/continuation was not performed completely; not a set of biological alternatives |
| Convention | An arbitrary content/index ordering declared separately from scientific preferences |
| Derivation | The identity of a computation’s parameters, code/numerical policy and upstream inputs |
| Payload binding | A digest of consumed source content; a different digest identifies changed evidence |
| Publication | An atomically committed typed product under its checked derivation |
| Marker | A workflow completion file written after checking publications; not scientific authority |
| Working-set model | A declared storage/cost estimate; not a memory reservation or a scientific cap |
| N, L, K | Sample count, site count and founder-row count in most methods |
| B, C, S, Q | Bins/blocks, contigs, model states/scoring samples and unordered pair states, as defined on each methods page |

Symbols are local to each stage. In cost expressions, lower-case sample/site
counts denote the active subset; iterations, frontier, alphabet and candidate
sizes are the mechanism’s actual dimensions or named limits. Check the unit’s
input contract before comparing costs from different models.
