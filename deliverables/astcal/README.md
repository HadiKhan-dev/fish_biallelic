# AstCal × AulStu handoff — preserved export

These are the readable handoff files exported on 30 July 2026, preserved without
rerunning the model. Start with [Tier-B annotated parentage](tier_B/pedigree_annotated.csv),
[evidence-labelled edges](tier_B/pedigree_edges_with_evidence.csv), or the
[pedigree graph](tier_B/pedigree_graph.svg). Family assignments/count matrices,
the ID crosswalk, and PLINK FAM exports are included. `source_manifest.json`
records original sources and hashes; `summary.json` retains export metadata.

Tier B is the supported-identity view. Tier C additionally uses the recorded
cross design to choose a near-complete operational hypothesis. The `complete/`
view adds one explicitly forced father edge beyond Tier C. Neither expanded
view should be mistaken for uniformly supported individual parentage.

Abstract G0 topology in these handoffs does not assign the sequenced G0 samples
as parents. Cohort membership, species identity, eligible-parent status and
inferred individual parentage are distinct quantities. There is no independent
individual-level trio truth: the preserved validation checks concern export
consistency and design compatibility, not real-data pedigree accuracy.
The original large run artifacts remain in local work storage.
