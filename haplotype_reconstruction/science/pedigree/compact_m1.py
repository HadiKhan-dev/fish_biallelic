"""Exact G4 M1 values as retained raw scores plus a counted default floor.

The genome gate/profile is already fixed. Never apply genome_profile to these
values again: the exact-sum floor is not idempotent. Dense expansion is explicit
and reserved for the reference path and migration checks.

Method: docs/methods/pedigree.md#candidate-panel
PE1–PE3 preserve this 3.5 method.
"""

import numpy as np

from ...contracts.pedigree import CompactM1


class CompactBuilder:
    def __init__(self, n, markers, eligible_counts):
        self.n = n
        self.markers = np.asarray(markers)
        self.counts = eligible_counts
        c = len(markers)
        self.best = np.full(n, -np.inf)
        self.cutoff = np.full(n, -np.inf)
        self.local = np.full((c, n), -np.inf)
        self.floors = self.local.copy()
        self.parents = [np.empty(0, np.int64) for _ in range(n)]
        self.raw = [np.empty((c, 0)) for _ in range(n)]

    def add(self, batch):
        for i, c in enumerate(batch["children"]):
            self.best[c] = batch["best"][i]
            self.cutoff[c] = batch["cutoff"][i]
            self.local[:, c] = batch["chromosome_best"][:, i]
            self.floors[:, c] = batch["floors"][:, i]
            parents = batch["retained"][i]
            self.parents[c] = parents
            self.raw[c] = batch["values"][:, i, parents].copy()

    def finish(self):
        indptr = np.r_[0, np.cumsum([len(x) for x in self.parents], dtype=np.int64)]
        return CompactM1(
            indptr,
            np.concatenate(self.parents),
            np.concatenate(self.raw, axis=1),
            self.best,
            self.cutoff,
            self.local,
            self.floors,
            self.markers,
            np.asarray(self.counts, np.int64),
        )
