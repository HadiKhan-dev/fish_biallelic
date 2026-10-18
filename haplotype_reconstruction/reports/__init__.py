"""The published tables of a run (OUTPUT/<stage>/...), written from the stages' records and products.

Reports never feed back into inference and never enter a scientific identity. The driver calls each writer after
the stage that feeds it (each returns the paths written):

  write_quality(output, store, contig, sample_ids)          quality/<contig>.samples.tsv, .segments.tsv
  write_reference_consistency(output, store, contigs, sample_ids, reference_rows, label=)
                                                            discovery/reference_consistency.csv
  write_painting(output, store, contig)                     painting/<contig>/*.tsv
  write_pedigree(output, product)                           pedigree/*.csv, evidence_calibration.json
  write_founder_refit(output, summaries)                    founder_refit/summary.csv
  write_family_phase(output, summaries)                     family_phase/summary.csv
  write_recombination(output, store, contigs)               recombination/...
  write_truth_pedigree(output, manifest)                    simulation/truth_pedigree.csv
  write_ties(output, store, contigs, pedigree_events)       ties/decisions.csv, summary.json (when the run
                                                            completes)
"""
from .painting import write_painting
from .pedigree import write_pedigree
from .quality import write_quality
from .recombination import write_recombination
from .reference_consistency import write_reference_consistency
from .tables import write_family_phase, write_founder_refit, write_truth_pedigree
from .ties import write_ties

__all__ = ["write_family_phase", "write_founder_refit", "write_painting", "write_pedigree", "write_quality",
           "write_recombination", "write_reference_consistency", "write_ties", "write_truth_pedigree"]
