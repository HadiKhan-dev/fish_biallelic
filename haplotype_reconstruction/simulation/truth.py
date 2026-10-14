"""Simulation truth of one chromosome, for downstream-only evaluation.

Inference never reads this record. The simulation writes it next to the
chromosome's simulated reads, once, while the truth arrays are resident.
"""
from __future__ import annotations

import numpy as np


TRUTH = "simulation/truth"
# Run-file record of a simulation: sample names, contigs, seed, true pedigree, state.
SIMULATION = "simulation/global"
FIELDS = ("truth_founder_haplotypes", "truth_painting", "truth_alleles", "truth_crossovers")


def save_truth(file, sample_ids, positions, source, *, nthreads=None):
    """Store the truth fields of ``source`` with their sample and site axes."""
    file.put(TRUTH, dict(sample_ids=tuple(map(str, sample_ids)), positions=np.asarray(positions),
                         **{key: source[key] for key in FIELDS}), nthreads=nthreads)


def load_truth(file, sample_ids, *, nthreads=None):
    """Return ({field: truth}, positions), checking the sample axis."""
    payload = file.get(TRUTH, nthreads=nthreads)
    if tuple(payload["sample_ids"]) != tuple(map(str, sample_ids)):
        raise ValueError(f"{file.path}: simulation truth sample axis differs")
    return {key: payload[key] for key in FIELDS}, np.asarray(payload["positions"])
