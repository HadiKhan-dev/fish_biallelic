# Tropheops handoff — preserved export

These are the readable handoff files exported on 24 July 2026, preserved without
rerunning or silently changing the inferred pedigree. Start with
[Tier-B annotated parentage](tier_B/pedigree_annotated.csv), its
[evidence-labelled edges](tier_B/pedigree_edges_with_evidence.csv), or the
[pedigree graph](tier_B/pedigree_graph.svg). The ID crosswalk and PLINK FAM files
support exchange with other tools. `source_manifest.json` records source files
and hashes; `summary.json` retains the original export metadata.

Important historical limitation: these exports contain G0-to-F1 inferred seed
edges under an earlier interpretation of the sequenced references. That is
**not the current pipeline's eligibility policy** and is not established
biological parentage. Some sequenced references are outside the pedigree. Do
not use these G0 edges as training truth or automatically admit those samples
as current parent candidates.

Tier A is stricter than Tier B. The `complete/` view includes leading hypotheses
below Tier B and three forced-completion edges; it is not a fully supported
pedigree. All tiers measure internal evidence/stability, not accuracy against
individual breeding records. No individual-level trio ground truth is claimed.
The original large run artifacts remain in local work storage.
